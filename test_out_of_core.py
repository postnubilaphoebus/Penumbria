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
from matching import matching
from postprocess import (
    TuningScoreError,
    objective,
    score_complete_volume,
    tuning_iou_thresholds,
    watershed_inference,
    watershed_inference_chunked,
)
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



# ───────────────────────── Optuna tuning scores whole validation volumes ─────────────────────────

TUNING_PARAMS = dict(gaussian_smoothing=False, h=0.3, bg=0.1, c=0.4, simple_thresholding=False)
ENOUGH_RAM = 8 * 1024 ** 3  # makes the global (non-tiled) watershed the one chosen, whatever this machine has free


def cell_labels(shape, boxes):
    labels = np.zeros(shape, np.int32)
    for cell, box in enumerate(boxes, start=1):
        labels[box] = cell
    return labels


def heat_map(labels, seed=3):
    """A noisy heat map of the labels: ~0.9 inside cells, <= 0.05 elsewhere (below every background threshold)."""
    rng = np.random.default_rng(seed)
    return (0.9 * (labels > 0) + 0.05 * rng.random(labels.shape)).astype(np.float32)


def store(directory, name, array, chunks):
    """The array as a disk-backed OME-Zarr volume with the given (small) storage chunks."""
    path = Path(directory) / f"{name}.ome.zarr"
    _, zarr_array = create_ome_zarr(path, array.shape, array.dtype, chunks=chunks, overwrite=True)
    zarr_array[:] = array
    return ChunkedVolume(path)


def reference_score(heat, labels, thresholds=None, **overrides):
    """The tuning metric computed by hand on ONE complete volume: flood it whole, then match at every threshold."""
    params = {**TUNING_PARAMS, **overrides}
    prediction = watershed_inference(
        heat, padding=None, minimum_cell_size=9, h=params["h"], cell_confidence_minimum=params["c"],
        background_threshold=params["bg"], gaussian_smoothing=params["gaussian_smoothing"],
        simple_thresholding=params["simple_thresholding"],
    )
    thresholds = np.arange(0.1, 1.0, 0.1) if thresholds is None else thresholds
    return sum(matching(labels, prediction, t).accuracy for t in thresholds) / len(thresholds)


class RecordingMatching:
    """Stands in for postprocess.matching: records what the metric is shown, then calls the real one."""

    def __init__(self):
        self.calls = []

    def __call__(self, y_true, y_pred, *args, **kwargs):
        self.calls.append(dict(
            true_shape=y_true.shape, pred_shape=y_pred.shape,
            true_objects=len(np.unique(y_true[y_true > 0])), pred_objects=len(np.unique(y_pred[y_pred > 0])),
        ))
        return matching(y_true, y_pred, *args, **kwargs)


class GlobalTuningScoreTests(unittest.TestCase):
    def setUp(self):
        self.budget = mock.patch.object(postprocess, "watershed_memory_budget_bytes", return_value=ENOUGH_RAM)
        self.budget.start()
        self.addCleanup(self.budget.stop)

    def four_cells(self):
        labels = cell_labels((24, 24, 24), [(slice(z, z + 6), slice(y, y + 6), slice(x, x + 6))
                                            for z, y, x in [(3, 3, 3), (3, 14, 12), (13, 4, 14), (14, 15, 3)]])
        return heat_map(labels), labels

    def test_objective_equals_the_direct_whole_volume_reference(self):
        heat, labels = self.four_cells()
        reference = reference_score(heat, labels)
        self.assertGreater(reference, 0.9)
        trial = optuna.trial.FixedTrial(TUNING_PARAMS)
        self.assertAlmostEqual(objective(trial, [heat], [labels], [None], 3), reference, places=12)
        with tempfile.TemporaryDirectory() as directory:  # the same volumes read back from disk
            stored_heat, stored_labels = store(directory, "heat", heat, (8, 8, 8)), store(directory, "labels", labels, (8, 8, 8))
            self.assertAlmostEqual(
                objective(trial, [stored_heat], [stored_labels], [None], 3, scratch_dir=directory), reference, places=12)

    def test_the_reference_holds_with_smoothing_and_simple_thresholding_too(self):
        heat, labels = self.four_cells()
        for overrides in (dict(gaussian_smoothing=True), dict(simple_thresholding=True),
                          dict(gaussian_smoothing=True, simple_thresholding=True, h=0.2)):
            with self.subTest(overrides):
                params = {**TUNING_PARAMS, **overrides}
                self.assertAlmostEqual(
                    objective(optuna.trial.FixedTrial(params), [heat], [labels], [None], 3),
                    reference_score(heat, labels, **overrides), places=12)

    def test_all_thresholds_in_one_matching_call_equal_one_call_per_threshold(self):
        truth = cell_labels((20, 20, 20), [(slice(2, 8), slice(2, 8), slice(2, 8)),
                                           (slice(10, 17), slice(10, 17), slice(10, 17)),
                                           (slice(2, 9), slice(12, 18), slice(3, 9))])
        shifted = np.roll(truth, (1, 2, 0), axis=(0, 1, 2))  # imperfect overlaps: accuracy changes with the threshold
        shifted[truth == 3] = 0
        thresholds = np.arange(0.1, 1.0, 0.1)
        per_threshold = [matching(truth, shifted, t).accuracy for t in thresholds]
        self.assertGreater(len(set(per_threshold)), 1)
        self.assertEqual(score_complete_volume(truth, shifted, thresholds), sum(per_threshold) / len(per_threshold))

    def test_iou_thresholds_and_2d_scoring_are_unchanged(self):
        np.testing.assert_allclose(tuning_iou_thresholds(3), np.arange(0.1, 1.0, 0.1))
        np.testing.assert_allclose(tuning_iou_thresholds(2), np.arange(0.5, 1.0, 0.05))
        self.assertEqual((len(tuning_iou_thresholds(3)), len(tuning_iou_thresholds(2))), (9, 10))
        labels = cell_labels((40, 40), [(slice(4, 12), slice(4, 12)), (slice(20, 30), slice(22, 32))])
        heat = heat_map(labels)
        self.assertAlmostEqual(
            objective(optuna.trial.FixedTrial(TUNING_PARAMS), [heat], [labels], [None], 2),
            reference_score(heat, labels, thresholds=np.arange(0.5, 1.0, 0.05)), places=12)

    def test_false_positives_outside_the_old_tuning_windows_lower_the_score(self):
        # 1700 planes is more than eight 192-plane windows: the old heuristic cut windows around the cells out of it.
        labels = cell_labels((1700, 8, 8), [(slice(10, 14), slice(2, 6), slice(2, 6)),
                                            (slice(1690, 1694), slice(2, 6), slice(2, 6))])
        clean = heat_map(labels)
        spurious = clean.copy()
        spurious[900:904, 2:6, 2:6] = 0.9  # a false positive far from both cells and from both windows
        trial = optuna.trial.FixedTrial(TUNING_PARAMS)
        windows = ([spurious[:192], spurious[-192:]], [labels[:192], labels[-192:]], [None, None])
        self.assertAlmostEqual(objective(trial, *windows, 3), 1.0, places=12)  # what window scoring saw
        self.assertAlmostEqual(objective(trial, [clean], [labels], [None], 3), 1.0, places=12)
        # ... and the complete volume: 2 TP, 1 FP, 0 FN at every threshold
        self.assertAlmostEqual(objective(trial, [spurious], [labels], [None], 3), 2 / 3, places=12)

    def test_a_volume_beyond_the_old_window_threshold_is_scored_complete_from_disk(self):
        labels = cell_labels((1700, 8, 8), [(slice(10, 14), slice(2, 6), slice(2, 6)),
                                            (slice(1690, 1694), slice(2, 6), slice(2, 6))])
        heat = heat_map(labels)
        heat[900:904, 2:6, 2:6] = 0.9
        recorder = RecordingMatching()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(postprocess, "matching", recorder):
            stored_heat = store(directory, "heat", heat, (64, 8, 8))
            stored_labels = store(directory, "labels", labels, (64, 8, 8))
            score = objective(optuna.trial.FixedTrial(TUNING_PARAMS), [stored_heat], [stored_labels], [None], 3,
                              scratch_dir=directory)
        self.assertAlmostEqual(score, 2 / 3, places=12)
        self.assertEqual(len(recorder.calls), 1)  # one score for the one volume ...
        self.assertEqual(recorder.calls[0], dict(true_shape=(1700, 8, 8), pred_shape=(1700, 8, 8),
                                                 true_objects=2, pred_objects=3))  # ... of everything in it

    def test_objects_crossing_storage_chunks_are_one_complete_instance(self):
        labels = cell_labels((24, 24, 24), [(slice(5, 11), slice(5, 11), slice(5, 11)),  # crosses the chunk planes at 8
                                            (slice(15, 22), slice(14, 21), slice(2, 20))])
        heat = heat_map(labels)
        recorder = RecordingMatching()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(postprocess, "matching", recorder):
            score = objective(optuna.trial.FixedTrial(TUNING_PARAMS),
                              [store(directory, "heat", heat, (8, 8, 8))],
                              [store(directory, "labels", labels, (8, 8, 8))], [None], 3, scratch_dir=directory)
        self.assertAlmostEqual(score, 1.0, places=12)
        self.assertEqual(recorder.calls, [dict(true_shape=(24, 24, 24), pred_shape=(24, 24, 24),
                                               true_objects=2, pred_objects=2)])

    def test_objects_crossing_watershed_tiles_are_one_complete_instance(self):
        zz, yy, xx = np.mgrid[:32, :40, :40]
        heat = np.maximum(
            np.exp(-((zz - 12) ** 2 + (yy - 17) ** 2 + (xx - 15) ** 2) / 30.0),
            np.exp(-((zz - 21) ** 2 + (yy - 25) ** 2 + (xx - 27) ** 2) / 30.0),
        ).astype(np.float32)
        params = dict(TUNING_PARAMS, simple_thresholding=True)
        labels = watershed_inference(heat, padding=None, minimum_cell_size=9, h=params["h"],
                                     cell_confidence_minimum=params["c"], background_threshold=params["bg"],
                                     gaussian_smoothing=False, simple_thresholding=True).astype(np.int32)
        self.assertEqual(int(labels.max()), 2)
        real_chunked = postprocess.watershed_inference_chunked
        full_shape = heat.shape

        def small_tiles(prediction, output_path, **kwargs):  # tiles of 16 x 24 x 24: both cells cross a tile border
            return real_chunked(prediction, output_path, **{**kwargs, "tile_shape": (16, 24, 24), "halo": 8})

        estimates = iter([10 ** 10])  # the whole volume does not fit (first question) ...

        def estimate(shape, **kwargs):
            return next(estimates, 1)  # ... but all the tiles do

        recorder = RecordingMatching()
        trial = optuna.trial.FixedTrial(params)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(postprocess, "matching", recorder), \
                mock.patch.object(postprocess, "estimate_watershed_memory_bytes", side_effect=estimate), \
                mock.patch.object(postprocess, "watershed_inference_chunked", side_effect=small_tiles) as chunked:
            stored_heat, stored_labels = store(directory, "heat", heat, (8, 8, 8)), store(directory, "labels", labels, (8, 8, 8))
            with self.assertWarns(RuntimeWarning):  # the approximate outer tiling announces itself
                score = objective(trial, [stored_heat], [stored_labels], [None], 3,
                                  volume_names=["tiled volume"], scratch_dir=directory)
        chunked.assert_called_once()
        self.assertAlmostEqual(score, 1.0, places=12)
        self.assertEqual(recorder.calls, [dict(true_shape=full_shape, pred_shape=full_shape,
                                               true_objects=2, pred_objects=2)])
        self.assertEqual(trial.user_attrs["approximate_tiled_watershed_volumes"], ["tiled volume"])

    def test_every_volume_counts_once_whatever_its_size_or_object_count(self):
        small_labels = cell_labels((12, 12, 12), [(slice(3, 9), slice(3, 9), slice(3, 9))])
        small_heat = heat_map(small_labels)  # 1 cell, found: score 1
        starts = [(z, y, x) for z in (3, 20) for y in (3, 20) for x in (3, 20)]
        big_labels = cell_labels((30, 30, 30), [(slice(z, z + 6), slice(y, y + 6), slice(x, x + 6)) for z, y, x in starts])
        big_heat = heat_map(big_labels)
        big_heat[big_labels > 4] = 0.0  # 8 cells, only the first 4 found: 4 TP, 4 FN, score 0.5
        trial = optuna.trial.FixedTrial(TUNING_PARAMS)
        self.assertAlmostEqual(reference_score(small_heat, small_labels), 1.0, places=12)
        self.assertAlmostEqual(reference_score(big_heat, big_labels), 0.5, places=12)
        with tempfile.TemporaryDirectory() as directory:
            disk_big = (store(directory, "big_heat", big_heat, (7, 11, 13)), store(directory, "big_labels", big_labels, (7, 11, 13)))
            for name, big in (("in memory", (big_heat, big_labels)), ("from disk", disk_big)):
                for order in (1, -1):
                    with self.subTest(name, order=order):
                        heats, truths = [list(pair)[::order] for pair in ((small_heat, big[0]), (small_labels, big[1]))]
                        score = objective(trial, heats, truths, [None, None], 3, scratch_dir=directory)
                        self.assertAlmostEqual(score, (1.0 + 0.5) / 2, places=12)
        pooled = (1 + 4) / (1 + 4 + 4)  # TP / (TP + FP + FN) over both volumes together
        self.assertNotAlmostEqual(score, pooled, places=2)

    def test_disk_backed_outputs_get_trial_and_volume_specific_paths_and_are_removed(self):
        heat, labels = self.four_cells()
        real_auto = postprocess.watershed_inference_auto
        paths = []

        def spy(prediction, output_path, **kwargs):
            paths.append(Path(output_path))
            return real_auto(prediction, output_path, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(postprocess, "watershed_inference_auto", side_effect=spy):
            volumes = [store(directory, f"heat{i}", heat, (8, 8, 8)) for i in range(2)]
            truths = [store(directory, f"labels{i}", labels, (8, 8, 8)) for i in range(2)]
            scratch = Path(directory) / "scratch"
            objective(optuna.trial.FixedTrial(TUNING_PARAMS), volumes, truths, [None, None], 3, scratch_dir=scratch)
            self.assertEqual(list(scratch.iterdir()), [])  # nothing accumulates over hundreds of trials
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0], paths[1])
        for index, path in enumerate(paths):
            self.assertTrue(path.parent.name.startswith("penumbria_tuning_trial0_"))
            self.assertEqual(path.parent.parent, scratch)
            self.assertIn(f"volume_{index}", path.name)

    def test_a_processing_failure_is_reported_with_trial_and_volume_not_scored_as_zero(self):
        heat, labels = self.four_cells()
        real = postprocess.watershed_inference
        calls = []

        def second_volume_breaks(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        trial = optuna.trial.FixedTrial(TUNING_PARAMS)
        with mock.patch.object(postprocess, "watershed_inference", side_effect=second_volume_breaks):
            with self.assertRaisesRegex(TuningScoreError, r"trial 0, validation volume 1 \(second\).*RuntimeError: boom") as caught:
                objective(trial, [heat, heat], [labels, labels], [None, None], 3, volume_names=["first", "second"])
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        # Optuna sees the exception: the study stops with a failed trial instead of recording a score.
        calls.clear()
        study = optuna.create_study(direction="maximize")
        previous = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        try:
            with mock.patch.object(postprocess, "watershed_inference", side_effect=second_volume_breaks):
                with self.assertRaises(TuningScoreError):
                    study.optimize(lambda t: objective(t, [heat, heat], [labels, labels], [None, None], 3), n_trials=3)
        finally:
            optuna.logging.set_verbosity(previous)
        self.assertEqual([t.state for t in study.trials], [optuna.trial.TrialState.FAIL])

    def test_out_of_memory_and_a_mismatched_shape_are_reported_and_leave_no_files_behind(self):
        heat, labels = self.four_cells()

        def runs_out_of_memory(prediction, output_path, **kwargs):
            Path(output_path).mkdir(parents=True)
            (Path(output_path) / "partial").write_text("half-written")
            raise MemoryError("tile bookkeeping")

        with tempfile.TemporaryDirectory() as directory:
            stored_heat, stored_labels = store(directory, "heat", heat, (8, 8, 8)), store(directory, "labels", labels, (8, 8, 8))
            scratch = Path(directory) / "scratch"
            with mock.patch.object(postprocess, "watershed_inference_auto", side_effect=runs_out_of_memory):
                with self.assertRaisesRegex(TuningScoreError, r"validation volume 0 \(big one\).*MemoryError: tile bookkeeping"):
                    objective(optuna.trial.FixedTrial(TUNING_PARAMS), [stored_heat], [stored_labels], [None], 3,
                              volume_names=["big one"], scratch_dir=scratch)
            self.assertEqual(list(scratch.iterdir()), [])
        with self.assertRaisesRegex(TuningScoreError, r"validation volume 1 \(cropped\).*shape.*\(24, 24, 24\).*\(24, 24, 23\)"):
            objective(optuna.trial.FixedTrial(TUNING_PARAMS), [heat, heat], [labels, labels[:, :, :23]], [None, None], 3,
                      volume_names=["fine", "cropped"])

    def test_inconsistent_inputs_are_rejected(self):
        heat, labels = self.four_cells()
        trial = optuna.trial.FixedTrial(TUNING_PARAMS)
        with self.assertRaises(ValueError):
            objective(trial, [heat, heat], [labels], [None, None], 3)  # zip would silently drop a volume
        with self.assertRaises(ValueError):
            objective(trial, [], [], [], 3)  # an average over no volumes is not a score


if __name__ == "__main__":
    unittest.main()
