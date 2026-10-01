import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch

from inference import sliding_window_inference
from postprocess import watershed_inference_chunked
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
            self.assertEqual(count, 2)
            self.assertEqual(len(np.unique(labels[:])) - 1, 2)


if __name__ == "__main__":
    unittest.main()
