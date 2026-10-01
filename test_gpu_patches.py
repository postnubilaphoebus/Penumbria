import itertools
import logging
import unittest
from unittest import mock

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter

import gpu_patched
from patch_plan import (GpuOutOfMemoryError, Patch, PatchBudget, _edges, call_or_oom, is_gpu_oom,
                        neighbours, plan_patches, run_to_fixed_point)

OOM = "CUDA Error CUDA_ERROR_OUT_OF_MEMORY: out of memory while calling malloc_async_impl"


def smooth_volume(shape, seed, levels=None):
    """A smooth random heat map with seeds on its peaks; `levels` quantises it into plateaus."""
    rng = np.random.default_rng(seed)
    heat = gaussian_filter(rng.random(shape).astype(np.float32), 2.0)
    heat = ((heat - heat.min()) / (heat.max() - heat.min())).astype(np.float32)
    if levels:
        heat = (np.round(heat * levels) / levels).astype(np.float32)
    seeds, _ = label((heat == maximum_filter(heat, size=5)) & (heat > 0.55))
    return heat, seeds.astype(np.int32)


def budget_for(max_labels_voxels):
    """A budget that allows patches of this many voxels in the heaviest (labels) phase."""
    return PatchBudget(max_labels_voxels * gpu_patched.LABELS_BYTES)


class PlanPatchesTests(unittest.TestCase):
    def test_cores_partition_the_volume_and_patches_fit_the_budget(self):
        for shape, limit in [((30, 34, 38), 3000), ((17, 5, 23), 200), ((10, 10, 10), 10_000)]:
            patches = plan_patches(shape, limit)
            covered = np.zeros(shape, int)
            for patch in patches:
                covered[patch.core] += 1
                self.assertLessEqual(patch.voxels, limit)
            self.assertTrue((covered == 1).all(), (shape, limit))

    def test_a_volume_that_fits_is_one_patch(self):
        (patch,) = plan_patches((30, 34, 38), 30 * 34 * 38)
        self.assertEqual(patch.core, patch.padded)

    def test_fewest_patches_matches_brute_force(self):
        def brute(shape, limit, halo=1):
            best = None
            for parts in itertools.product(*[range(1, size + 1) for size in shape]):
                edges = [_edges(size, k) for size, k in zip(shape, parts)]
                largest = 0
                for index in itertools.product(*[range(k) for k in parts]):
                    voxels = 1
                    for axis, i in enumerate(index):
                        low, high = int(edges[axis][i]), int(edges[axis][i + 1])
                        voxels *= min(shape[axis], high + halo) - max(0, low - halo)
                    largest = max(largest, voxels)
                if largest <= limit:
                    key = (parts[0] * parts[1] * parts[2], largest)
                    best = key if best is None else min(best, key)
            return best

        rng = np.random.default_rng(0)
        for _ in range(40):
            shape = tuple(int(v) for v in rng.integers(1, 9, 3))
            limit = int(rng.integers(27, 300))
            plan = plan_patches(shape, limit)
            self.assertEqual((len(plan), max(p.voxels for p in plan)), brute(shape, limit), (shape, limit))

    def test_budget_too_small_for_any_patch(self):
        with self.assertRaises(ValueError):
            plan_patches((10, 10, 10), 26)  # a single voxel plus its halo needs 27

    def test_neighbours_are_the_patches_whose_halo_reaches_in(self):
        patches = plan_patches((8, 8, 8), 5 ** 3)  # 2 x 2 x 2 patches
        self.assertEqual(len(patches), 8)
        self.assertTrue(all(len(others) == 7 for others in neighbours(patches)))  # halos reach the diagonals too


class OutOfMemoryPolicyTests(unittest.TestCase):
    """The policy needs no GPU: updates here raise the error Taichi raises."""

    def run_phase(self, budget, update, shape=(20, 20, 20), raw=10):
        run_to_fixed_point(shape, budget, update, phase="test", raw_bytes_per_voxel=raw)

    def test_oom_is_recognised(self):
        self.assertTrue(is_gpu_oom(RuntimeError(OOM)))
        self.assertTrue(is_gpu_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertFalse(is_gpu_oom(RuntimeError("something else")))

    def test_first_oom_logs_the_request_halves_and_carries_on(self):
        seen = []

        def update(patch):
            if patch.voxels > 5000:
                raise RuntimeError(OOM)
            seen.append(patch.voxels)
            return False

        budget = PatchBudget(bytes=8000 * 10)  # patches of up to 8000 voxels at 10 bytes each
        with self.assertLogs("penumbria.gpu", "WARNING") as logs:
            self.run_phase(budget, update)
        message = logs.output[0]
        first = plan_patches((20, 20, 20), 8000)[0]
        self.assertIn(f"{first.voxels:,} voxels", message)
        self.assertIn(f"requested {first.voxels * 10:,} bytes", message)
        self.assertIn("Halving", message)
        self.assertEqual(budget.bytes, 4000 * 10)
        self.assertTrue(seen and max(seen) <= 4000)  # the retry ran with patches of at most half the voxels

    def test_second_oom_right_after_the_halving_fails_loudly(self):
        calls = []

        def update(patch):
            calls.append(patch.voxels)
            raise RuntimeError(OOM)

        budget = PatchBudget(bytes=8000 * 10)
        with self.assertLogs("penumbria.gpu", "WARNING"), self.assertRaises(GpuOutOfMemoryError) as caught:
            self.run_phase(budget, update)
        self.assertEqual(len(calls), 2)  # the original patch, then one retry with half the voxels
        self.assertEqual(budget.bytes, 4000 * 10)  # halved exactly once, not indefinitely
        self.assertIn("halved once", str(caught.exception))
        self.assertIn("requested", str(caught.exception))

    def test_a_success_after_the_halving_allows_a_later_failure_one_more_halving(self):
        outcomes = iter(["oom", "ok", "oom", "ok"])

        def update(patch):
            if next(outcomes, "ok") == "oom":
                raise RuntimeError(OOM)
            return False

        budget = PatchBudget(bytes=8000 * 10)
        with self.assertLogs("penumbria.gpu", "WARNING"):
            self.run_phase(budget, update)
        self.assertEqual(budget.bytes, 2000 * 10)  # two separate failures, one halving each

    def test_other_errors_are_not_swallowed(self):
        def update(patch):
            raise RuntimeError("kernel launch failed")

        with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
            self.run_phase(PatchBudget(bytes=8000 * 10), update)

    def test_check_free_turns_a_shortfall_into_an_oom_without_the_driver_raising(self):
        patch = Patch((slice(0, 10),) * 3, (slice(0, 10),) * 3)
        budget = PatchBudget(bytes=10**9, overhead=1.5, free_bytes=lambda: 1_000)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            budget.check_free(patch, 10)  # needs 1000 voxels * 10 B * 1.5 = 15000 > 1000
        PatchBudget(bytes=10**9, free_bytes=lambda: 10**6).check_free(patch, 10)

    def test_changes_reach_the_neighbours(self):
        values = np.zeros((12, 12, 12), int)

        def update(patch):  # a changed patch hands its value to the next one: a plateau crossing the volume
            core = values[patch.core]
            before = core.copy()
            lower = values[patch.padded].max()
            core[...] = np.maximum(core, lower)
            if patch.core[0].start == 0 and patch.core[1].start == 0 and patch.core[2].start == 0:
                core[...] = 1
            values[patch.core] = core
            return not np.array_equal(before, core)

        self.run_phase(budget_for(300), update, shape=values.shape)
        self.assertTrue((values == 1).all())


class WholeVolumePathTests(unittest.TestCase):
    def test_oom_on_the_unpatched_run_halves_once_then_patches(self):
        budget = budget_for(30 * 34 * 38)
        calls = {"unpatched": 0}

        def unpatched():
            calls["unpatched"] += 1
            raise RuntimeError(OOM)

        with self.assertLogs("penumbria.gpu", "WARNING"):
            result = gpu_patched._unpatched_or_patched(
                (30, 34, 38), budget, gpu_patched.LABELS_BYTES, "test", unpatched, lambda: "patched")
        self.assertEqual((result, calls["unpatched"]), ("patched", 1))

    def test_oom_again_after_the_halving_fails_loudly(self):
        budget = budget_for(1000)  # still a single patch of the 64-voxel volume after halving

        def unpatched():
            raise RuntimeError(OOM)

        with self.assertLogs("penumbria.gpu", "WARNING"), self.assertRaises(GpuOutOfMemoryError):
            gpu_patched._unpatched_or_patched((4, 4, 4), budget, gpu_patched.LABELS_BYTES, "test",
                                              unpatched, lambda: "patched")


class PatchedGpuResultTests(unittest.TestCase):
    """Patched runs must give exactly the labels and h-dome of the unpatched GPU functions."""

    def assertWatershedExact(self, heat, seeds, background, limits):
        expected = gpu_patched.seeded_watershed_whole(heat, seeds, background)
        for limit in limits:
            self.assertGreater(len(plan_patches(heat.shape, limit)), 1)
            got = gpu_patched.seeded_watershed(heat, seeds, background, budget=budget_for(limit))
            self.assertTrue(np.array_equal(expected, got), f"limit {limit}: {(expected != got).sum()} voxels differ")
        return expected

    def test_random_volumes(self):
        for seed, shape in [(0, (26, 28, 30)), (7, (27, 25, 29))]:
            heat, seeds = smooth_volume(shape, seed)
            labels = self.assertWatershedExact(heat, seeds, 0.1, [3500, 1800])
            self.assertGreater(labels.max(), 5)

    def test_plateaus_across_patch_edges(self):
        for seed, levels in [(1, 3), (2, 6)]:  # few height levels: plateaus everywhere
            heat, seeds = smooth_volume((22, 24, 26), seed, levels)
            self.assertWatershedExact(heat, seeds, 0.1, [2500, 1500])

    def test_a_tie_on_a_long_plateau_between_two_seeds(self):
        heat = np.zeros((9, 9, 120), np.float32)
        heat[4, 4, :] = 0.5
        heat[4, 4, [0, 119]] = 0.9
        seeds = np.zeros(heat.shape, np.int32)
        seeds[4, 4, 0], seeds[4, 4, 119] = 1, 2
        labels = self.assertWatershedExact(heat, seeds, 0.1, [2500])
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})

    def test_one_seed_floods_a_plateau_spanning_many_patches(self):
        heat = np.full((20, 20, 20), 0.4, np.float32)
        heat[0, 0, 0] = 0.9
        seeds = np.zeros(heat.shape, np.int32)
        seeds[0, 0, 0] = 7
        labels = self.assertWatershedExact(heat, seeds, 0.1, [1500])
        self.assertTrue((labels == 7).all())

    def test_no_seeds_and_no_foreground(self):
        heat, seeds = smooth_volume((20, 20, 20), 5)
        self.assertFalse(self.assertWatershedExact(heat, np.zeros_like(seeds), 0.1, [1000]).any())
        self.assertFalse(self.assertWatershedExact(heat, seeds, 2.0, [1000]).any())

    def test_seed_scan_skips_patches_no_seed_can_reach(self):
        heat = np.zeros((20, 20, 40), np.float32)
        seeds = np.zeros(heat.shape, np.int32)
        heat[8:12, 8:12, 2:6] = 0.8
        heat[10, 10, 3] = 0.9
        seeds[10, 10, 3] = 5
        with self.assertLogs("penumbria.gpu", "INFO") as logs:
            labels = self.assertWatershedExact(heat, seeds, 0.1, [2000])
        scan = next(line for line in logs.output if "seed scan" in line)
        unreached, total = [int(word) for word in scan.replace("seed scan:", "").split() if word.isdigit()][:2]
        self.assertGreater(unreached, 0)
        self.assertLess(unreached, total)
        self.assertEqual(int((labels == 5).sum()), int((heat > 0.1).sum()))

    def test_a_basin_reaches_into_a_patch_that_has_no_seed(self):
        heat = np.zeros((10, 10, 60), np.float32)
        heat[4:6, 4:6, :] = np.linspace(0.9, 0.3, 60, dtype=np.float32)  # one long cell, one seed at its high end
        seeds = np.zeros(heat.shape, np.int32)
        seeds[4, 4, 0] = 3
        labels = self.assertWatershedExact(heat, seeds, 0.1, [700])
        self.assertTrue((labels[4:6, 4:6, :] == 3).all())

    def test_h_dome(self):
        heat, _ = smooth_volume((24, 26, 28), 0)
        expected = gpu_patched.h_dome_whole(heat, 0.2)
        for limit in (6000, 3000):
            got = gpu_patched.h_dome(heat, 0.2, budget=PatchBudget(limit * gpu_patched.LEVELS_BYTES))
            self.assertTrue(np.array_equal(expected, got))

    def test_unpatched_path_is_the_original_function(self):
        heat, seeds = smooth_volume((20, 20, 20), 3)
        with mock.patch.object(gpu_patched, "_watershed_patched") as patched:
            got = gpu_patched.seeded_watershed(heat, seeds, 0.1, budget=budget_for(10**7))
        patched.assert_not_called()
        self.assertTrue(np.array_equal(got, gpu_patched.seeded_watershed_whole(heat, seeds, 0.1)))

    def test_two_dimensional_input(self):
        heat, seeds = smooth_volume((1, 40, 50), 4)
        got = gpu_patched.seeded_watershed(heat[0], seeds[0], 0.1, budget=budget_for(10**7))
        self.assertEqual(got.shape, (40, 50))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
