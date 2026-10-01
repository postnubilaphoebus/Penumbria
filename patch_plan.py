"""Splitting a volume into the fewest patches that fit a memory budget, and running a
fixed-point computation over them. Pure Python/NumPy: no GPU code in here."""
from dataclasses import dataclass
import gc
import logging
import math

import numpy as np

log = logging.getLogger("penumbria.gpu")

MAX_VOXELS = 2**31 - 1  # Taichi indexes ndarrays with 32 bits


class GpuOutOfMemoryError(RuntimeError):
    """The GPU ran out of memory again right after the patch size was halved."""


def is_gpu_oom(error):
    text = str(error)
    return "CUDA_ERROR_OUT_OF_MEMORY" in text or "out of memory" in text.lower()


def call_or_oom(function, *args):
    """(result, None) on success, (None, message) if the GPU ran out of memory."""
    try:
        return function(*args), None
    except RuntimeError as error:
        if not is_gpu_oom(error):
            raise
        message = str(error)
    # Collected only now: while the exception is alive, its traceback still holds the GPU arrays.
    gc.collect()
    return None, message


@dataclass(frozen=True)
class Patch:
    core: tuple    # the voxels this patch is responsible for
    padded: tuple  # the core plus a halo of neighbouring voxels, clipped to the volume

    @property
    def voxels(self):
        return math.prod(axis.stop - axis.start for axis in self.padded)

    @property
    def core_in_padded(self):
        return tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(self.core, self.padded))


def _edges(size, parts):
    return np.arange(parts + 1) * size // parts


def _padded_extent(size, parts, halo):
    """Largest padded patch length along an axis cut into `parts` pieces."""
    edges = _edges(size, parts)
    return int((np.minimum(size, edges[1:] + halo) - np.maximum(0, edges[:-1] - halo)).max())


def _axis_options(size, halo):
    """(parts, padded length) for every number of parts that gives shorter patches than fewer parts do."""
    options = []
    for parts in range(1, size + 1):
        extent = _padded_extent(size, parts, halo)
        if not options or extent < options[-1][1]:
            options.append((parts, extent))
    return options


def plan_patches(shape, max_voxels, halo=1):
    """The fewest axis-aligned patches whose padded size is at most `max_voxels`.

    Cores partition the volume, with lengths differing by at most one voxel along
    an axis. Among plans with the fewest patches, the one with the smallest patches wins.
    """
    z_options, y_options, x_options = (_axis_options(size, halo) for size in shape)
    best = None
    for z_parts, z_extent in z_options:
        for y_parts, y_extent in y_options:
            plane = z_extent * y_extent
            if plane > max_voxels:
                continue
            for x_parts, x_extent in x_options:
                voxels = plane * x_extent
                if voxels <= max_voxels:
                    key = (z_parts * y_parts * x_parts, voxels)
                    if best is None or key < best[0]:
                        best = (key, (z_parts, y_parts, x_parts))
                    break  # more x parts only add patches
    if best is None:
        raise ValueError(f"A budget of {max_voxels} voxels is too small for any patch of a {tuple(shape)} volume")
    edges = [_edges(size, parts) for size, parts in zip(shape, best[1])]
    patches = []
    for z in range(best[1][0]):
        for y in range(best[1][1]):
            for x in range(best[1][2]):
                bounds = [(edges[axis][i], edges[axis][i + 1]) for axis, i in enumerate((z, y, x))]
                core = tuple(slice(low, high) for low, high in bounds)
                padded = tuple(slice(max(0, low - halo), min(size, high + halo))
                               for (low, high), size in zip(bounds, shape))
                patches.append(Patch(core, padded))
    return patches


def neighbours(patches):
    """For every patch, the indices of the other patches whose halo reaches into its core."""
    cores = np.array([[(axis.start, axis.stop) for axis in patch.core] for patch in patches])
    padded = np.array([[(axis.start, axis.stop) for axis in patch.padded] for patch in patches])
    reaches = np.all((cores[:, None, :, 0] < padded[None, :, :, 1])
                     & (padded[None, :, :, 0] < cores[:, None, :, 1]), axis=2)
    np.fill_diagonal(reaches, False)
    return [np.flatnonzero(row).tolist() for row in reaches]


@dataclass
class PatchBudget:
    """GPU memory the patches may use, and the policy for running out of it anyway."""
    bytes: int
    overhead: float = 1.0      # measured allocator overhead on top of the raw array sizes
    free_bytes: object = None  # callable returning the GPU memory free right now
    recovering: bool = False   # True from an out-of-memory halving until a patch succeeds again

    def limit_voxels(self, raw_bytes_per_voxel):
        return max(1, min(MAX_VOXELS, int(self.bytes / (raw_bytes_per_voxel * self.overhead))))

    def check_free(self, patch, raw_bytes_per_voxel):
        """Raise the out-of-memory error up front if the GPU has less free memory than the patch needs.

        Some systems (Windows with the NVIDIA system-memory fallback, notably) let allocations beyond the
        GPU's memory succeed by spilling into system RAM, thousands of times slower, instead of failing.
        Checking the free memory before each patch makes the out-of-memory policy independent of that.
        """
        if self.free_bytes is None:
            return
        needed, free = patch.voxels * raw_bytes_per_voxel * self.overhead, self.free_bytes()
        if free < needed:
            raise RuntimeError(f"out of memory (checked before allocating): the patch needs about "
                               f"{needed:,.0f} bytes including allocator overhead, the GPU has {free:,} free")

    def halve_after_oom(self, patch, phase, raw_bytes_per_voxel):
        """Log what the failed patch asked for and halve the budget; fail if that was already a retry."""
        requested = patch.voxels * raw_bytes_per_voxel
        free = "" if self.free_bytes is None else f", {self.free_bytes() / 2**30:.2f} GiB free on the GPU now"
        extent = " x ".join(str(axis.stop - axis.start) for axis in patch.padded)
        what = (f"GPU out of memory in '{phase}': the patch of {extent} = {patch.voxels:,} voxels "
                f"requested {requested:,} bytes ({requested / 2**30:.2f} GiB){free}")
        if self.recovering:
            raise GpuOutOfMemoryError(
                f"{what}. This happened right after the patch size was halved once, so it is not just "
                "other programs briefly using the GPU. Close GPU programs, check for a memory leak, "
                "or use a smaller volume.")
        self.bytes //= 2
        self.recovering = True
        log.warning("%s. Halving the patch size to at most %s voxels and retrying once.",
                    what, f"{self.limit_voxels(raw_bytes_per_voxel):,}")


def run_to_fixed_point(shape, budget, update, *, phase, raw_bytes_per_voxel):
    """Call `update(patch)` on every patch, then again on the neighbours of every patch it changed,
    until no patch changes. `update` returns whether it changed its core.

    Each patch sees the latest values of its halo, so neighbours exchange information through
    the arrays that `update` reads and writes. If the GPU runs out of memory the budget is halved
    (see PatchBudget) and the phase carries on with smaller patches; nothing already computed is lost.
    """
    while True:
        patches = plan_patches(shape, budget.limit_voxels(raw_bytes_per_voxel))
        adjacent = neighbours(patches)
        log.info("%s: %d patch(es) of at most %s voxels", phase, len(patches),
                 f"{max(patch.voxels for patch in patches):,}")
        dirty, sweep, restart = set(range(len(patches))), 0, False
        while dirty and not restart:
            order = sorted(dirty, reverse=sweep % 2 == 1)  # alternate direction: information travels fast both ways
            dirty, sweep = set(), sweep + 1
            for index in order:
                changed, oom = call_or_oom(update, patches[index])
                if oom is not None:
                    budget.halve_after_oom(patches[index], phase, raw_bytes_per_voxel)
                    restart = True
                    break
                budget.recovering = False
                if changed:
                    dirty.update(adjacent[index])
        if not restart:
            return
