import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import optuna
import tifffile
import torch

from inference import sliding_window_inference
import postprocess
from postprocess import objective, prepare_tuning_patches, watershed_inference, watershed_inference_chunked
from utils import preprocess_0_1
from volume_io import (
    ChunkedVolume,
    create_ome_zarr,
    imagej_label_dtype,
    open_cached_volume,
)


class IdentityModel(torch.nn.Module):
    def forward(self, values, *args):
        return values, None


class OutOfCoreTests(unittest.TestCase):
    def test_exact_global_normalisation_matches_numpy(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(9, 11, 13)).astype(np.float32)
        source[0] = -3.0
        source[-1] = 8.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.tif"
            tifffile.imwrite(path, source, metadata={"axes": "ZYX"})
            volume = open_cached_volume(path, normalize=True, low_percentile=1.0, high_percentile=99.9)
            expected = preprocess_0_1(source, low_clip=1.0, high_clip=99.9)
            actual = volume[:]
            self.assertTrue(np.allclose(actual, expected, rtol=0, atol=2e-7))
            expected_percentiles = np.percentile(source, [1.0, 99.9])
            self.assertEqual(volume.normalization.low_value, expected_percentiles[0])
            self.assertEqual(volume.normalization.high_value, expected_percentiles[1])

    def test_lazy_patch_padding_matches_full_numpy_padding(self):
        source = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.tif"
            tifffile.imwrite(path, source, metadata={"axes": "ZYX"})
            volume = open_cached_volume(path, normalize=False)
            patch = volume.read_patch((0, 0, 0), (4, 4, 4), pad_mode="reflect")
            padded = np.pad(source, ((2, 2),) * 3, mode="reflect")
            self.assertTrue(np.array_equal(patch, padded[0:4, 0:4, 0:4]))

            integer_volume = open_cached_volume(
                path, cache_dir=Path(directory) / "integer_cache", storage_dtype=np.int32, read_dtype=np.int64
            )
            integer_patch = integer_volume.read_patch(
                (0, 0, 0), (4, 4, 4), pad_mode="constant", constant_values=-100
            )
            integer_padded = np.pad(source.astype(np.int64), ((2, 2),) * 3, mode="constant", constant_values=-100)
            self.assertTrue(np.array_equal(integer_patch, integer_padded[0:4, 0:4, 0:4]))

    def test_disk_backed_inference_matches_identity(self):
        source = np.linspace(0, 1, 7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "image.tif"
            tifffile.imwrite(source_path, source, metadata={"axes": "ZYX"})
            volume = open_cached_volume(source_path, normalize=False)
            outputs, names, padding = sliding_window_inference(
                IdentityModel(), [volume], False, None, None, torch.device("cpu"),
                0.0, 1.0, directory, ["identity"], False,
                patch_based_norm=False, tta=False, image_dim=(4, 4, 4),
                keep_size=(2, 2, 2), step_size=(2, 2, 2), save_files=True,
            )
            self.assertEqual(names, ["identity"])
            self.assertEqual(padding, [None])
            self.assertIsInstance(outputs[0], ChunkedVolume)
            self.assertTrue(np.allclose(outputs[0][:], source, atol=2e-6))
            imagej_path = Path(directory) / "preds" / "identity_inference_output.tif"
            with tifffile.TiffFile(imagej_path) as imagej_tiff:
                self.assertTrue(imagej_tiff.is_imagej)
                self.assertFalse(imagej_tiff.is_bigtiff)
                self.assertEqual(imagej_tiff.series[0].dtype, np.dtype(np.float32))
                self.assertTrue(np.allclose(imagej_tiff.asarray(), source, atol=2e-6))

    def test_patch_based_normalisation_scales_each_patch_on_its_own(self):
        rng = np.random.default_rng(1)
        dark, bright = rng.random((8, 8, 8)), 10 + 5 * rng.random((8, 8, 8))
        source = np.concatenate([dark, bright], axis=2).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "image.tif"
            tifffile.imwrite(source_path, source, metadata={"axes": "ZYX"})
            volume = open_cached_volume(source_path, normalize=False)

            def infer(patch_based_norm):
                outputs, _, _ = sliding_window_inference(
                    IdentityModel(), [volume], False, None, None, torch.device("cpu"),
                    0.0, 1.0, directory, ["tiles"], False,
                    patch_based_norm=patch_based_norm, tta=False, image_dim=(8, 8, 8),
                    keep_size=(8, 8, 8), step_size=(8, 8, 8), save_files=False,
                )
                return outputs[0][:]

            expected = np.concatenate([preprocess_0_1(source[..., :8], 1.0, 99.9),
                                       preprocess_0_1(source[..., 8:], 1.0, 99.9)], axis=2)
            self.assertTrue(np.allclose(infer(True), expected, atol=2e-6))
            unnormalised = infer(False)  # the identity model passes the raw values on, clipped to [0, 1]
            self.assertTrue(np.allclose(unnormalised[..., :8], source[..., :8], atol=2e-6))
            self.assertTrue((unnormalised[..., 8:] == 1.0).all())

    def tuning_inputs(self, directory, labels):
        """Cached prediction (a noisy heat map of the labels) and label volumes for prepare_tuning_patches."""
        rng = np.random.default_rng(3)
        prediction = (0.9 * (labels > 0) + 0.05 * rng.random(labels.shape)).astype(np.float32)
        paths = {}
        for name, array in (("prediction", prediction), ("labels", labels.astype(np.int32))):
            paths[name] = Path(directory) / f"{name}.tif"
            tifffile.imwrite(paths[name], array, metadata={"axes": "ZYX"})
        heat = open_cached_volume(paths["prediction"], storage_dtype=np.float32)
        integer = open_cached_volume(paths["labels"], storage_dtype=np.int32, read_dtype=np.int32, is_label=True)
        return prediction, heat, integer

    def test_tuning_windows_stay_inside_the_volume_and_match_the_labels(self):
        labels = np.zeros((700, 12, 12), np.int32)  # more than 3 windows of 192 planes: windows are cheaper
        for cell, z in enumerate((10, 350, 690), start=1):
            labels[z - 2:z + 2, 4:8, 4:8] = cell
        with tempfile.TemporaryDirectory() as directory:
            prediction, heat, integer = self.tuning_inputs(directory, labels)
            windows, window_labels = prepare_tuning_patches([heat], [integer], [heat], (4, 4, 4), max_patches_per_image=3)
        self.assertEqual(len(windows), 3)  # one per cell, no two alike
        for window, window_label in zip(windows, window_labels):
            self.assertEqual(window.shape, (192, 12, 12))
            start = next(z for z in range(700 - 192 + 1) if np.array_equal(prediction[z:z + 192], window))
            self.assertTrue(np.array_equal(labels[start:start + 192], window_label))  # same voxels, no padding
        self.assertEqual({int(c) for window_label in window_labels for c in np.unique(window_label)} - {0}, {1, 2, 3})

    def test_a_volume_smaller_than_the_window_is_tuned_whole_and_scores_like_the_whole_image(self):
        labels = np.zeros((24, 24, 24), np.int32)
        for cell, (z, y, x) in enumerate([(3, 3, 3), (3, 14, 12), (13, 4, 14), (14, 15, 3)], start=1):
            labels[z:z + 6, y:y + 6, x:x + 6] = cell
        params = dict(gaussian_smoothing=False, h=0.3, bg=0.1, c=0.4, simple_thresholding=False)
        with tempfile.TemporaryDirectory() as directory:
            prediction, heat, integer = self.tuning_inputs(directory, labels)
            windows, window_labels = prepare_tuning_patches([heat], [integer], [heat], (64, 64, 64))
        self.assertEqual(len(windows), 1)
        self.assertTrue(np.array_equal(windows[0], prediction) and np.array_equal(window_labels[0], labels))
        trial = optuna.trial.FixedTrial(params)
        whole = objective(trial, [prediction], [labels], [None], 3)
        self.assertGreater(whole, 0.9)  # clean cells, so a good score ...
        self.assertEqual(objective(trial, windows, window_labels, [None], 3), whole)  # ... and the same one

    def test_an_image_is_tuned_whole_unless_the_windows_are_cheaper(self):
        labels = np.zeros((230, 12, 12), np.int32)  # 230 planes: less than 8 windows of 192 planes
        labels[100:104, 4:8, 4:8] = 1
        with tempfile.TemporaryDirectory() as directory:
            prediction, heat, integer = self.tuning_inputs(directory, labels)
            windows, window_labels = prepare_tuning_patches([heat], [integer], [heat], (4, 4, 4))
        self.assertEqual(len(windows), 1)
        self.assertTrue(np.array_equal(windows[0], prediction) and np.array_equal(window_labels[0], labels))

    def test_small_images_use_skimage_and_give_the_gpu_result(self):
        labels = np.zeros((24, 24, 24), np.int32)
        for cell, (z, y, x) in enumerate([(3, 3, 3), (3, 14, 12), (13, 4, 14), (14, 15, 3)], start=1):
            labels[z:z + 6, y:y + 6, x:x + 6] = cell
        heat = (0.9 * (labels > 0)).astype(np.float32)
        settings = dict(padding=None, h=0.3, cell_confidence_minimum=0.4, background_threshold=0.1,
                        gaussian_smoothing=False)
        with mock.patch.object(postprocess, "seeded_watershed", side_effect=AssertionError("GPU watershed used")):
            small = watershed_inference(heat, **settings)  # 13,824 voxels: below the threshold
        gpu = watershed_inference(heat, skimage_below_voxels=0, **settings)
        self.assertEqual(int(small.max()), 4)
        self.assertTrue(np.array_equal(small, gpu))

    def test_imagej_label_dtype_preserves_ids(self):
        self.assertEqual(imagej_label_dtype(162), np.dtype(np.uint16))
        self.assertEqual(imagej_label_dtype(70_000), np.dtype(np.float32))
        with self.assertRaises(ValueError):
            imagej_label_dtype(2**24 + 1)

    def test_chunked_watershed_merges_across_tiles(self):
        zz, yy, xx = np.mgrid[:32, :40, :40]
        prediction = np.maximum(
            np.exp(-((zz - 12) ** 2 + (yy - 17) ** 2 + (xx - 15) ** 2) / 30.0),
            np.exp(-((zz - 21) ** 2 + (yy - 25) ** 2 + (xx - 27) ** 2) / 30.0),
        ).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "prediction.ome.zarr"
            _, array = create_ome_zarr(prediction_path, prediction.shape, np.float32, overwrite=True)
            array[:] = prediction
            with mock.patch.object(postprocess, "seeded_watershed", wraps=postprocess.seeded_watershed) as gpu:
                labels, count = watershed_inference_chunked(
                    ChunkedVolume(prediction_path),
                    Path(directory) / "labels.ome.zarr",
                    minimum_cell_size=10,
                    h=0.6,
                    cell_confidence_minimum=0.7,
                    background_threshold=0.08,
                    gaussian_smoothing=False,
                    simple_thresholding=True,
                    tile_shape=(16, 24, 24),
                    halo=8,
                )
            gpu.assert_called()  # tiles are small, but they must all be flooded by the same (GPU) implementation
            self.assertEqual(count, 2)
            self.assertEqual(len(np.unique(labels[:])) - 1, 2)


if __name__ == "__main__":
    unittest.main()
