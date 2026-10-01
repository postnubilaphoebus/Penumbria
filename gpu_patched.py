"""GPU h-dome and seeded watershed that stay inside a GPU memory budget.

`h_dome` and `seeded_watershed` here have the signatures of the ones in gpu_morphology and
gpu_watershed and return the same arrays. A volume that fits in 70% of the free GPU memory
goes through those functions unchanged. A larger one is cut into the fewest patches that
fit, and the result is still exactly what the unpatched run would give:

Both operations are fixed points of monotone steps that only look at the six neighbours of
a voxel. Each patch is padded with a one-voxel halo, solved on the GPU with the latest
values of that halo, and written back; patches next to a changed patch are solved again
until a full pass changes nothing. Where a basin or a plateau crosses a patch edge, the
information crosses it through the halo, so there are no seams.

The watershed runs in three such phases, each reading what the previous one settled:
  1. flood levels   reconstruction by dilation: the highest level at which each voxel is
                    connected to a seed (the 'shave' of gpu_watershed);
  2. plateau steps  breadth-first distance of each plateau voxel to the plateau's exit;
  3. labels         every voxel points at its highest neighbour (or, on a plateau, at a
                    neighbour one step closer to the exit) and takes that neighbour's label.

Patches that no seed can reach never touch the GPU: that is the seed scan. A patch without
a seed of its own is not simply background, though, because a basin from a neighbouring
patch can extend into it; the flood levels decide.
"""
import logging

import numpy as np
import taichi as ti

from gpu_memory import WATERSHED_BYTES_PER_VOXEL, free_gpu_bytes, watershed_budget
from gpu_morphology import (F32, I32, LIST, NEIGHBOURS, frees_gpu_memory_here, h_dome as h_dome_whole,
                            index_of, inside, reconstruct_by_dilation_, to_device, until_stable, voxel_of)
from gpu_watershed import FLAT, UNLABELED, UNRESOLVED, seeded_watershed as seeded_watershed_whole
from patch_plan import call_or_oom, plan_patches, run_to_fixed_point

log = logging.getLogger("penumbria.gpu")

# Bytes per voxel that each phase asks the GPU for at its peak.
LEVELS_BYTES = 12    # marker, ceiling, list of voxels still below their ceiling
DISTANCE_BYTES = 8   # levels, plateau distance
LABELS_BYTES = WATERSHED_BYTES_PER_VOXEL  # heat, seeds, levels, distance, labels, parent


# ─────────────────────────────── kernels ───────────────────────────────

@ti.kernel
def _relax_plateau_distance(levels: F32, distance: I32, changed: LIST):
    # distance[v] = 1 + the smallest distance among equal-level neighbours: breadth-first
    # distance to the nearest voxel that has a higher neighbour (distance 0). In place is
    # safe: every value read is still an upper bound.
    for z, y, x in levels:
        current = distance[z, y, x]
        if current > 0:
            best = current
            for dz, dy, dx in ti.static(NEIGHBOURS):
                if inside(levels, z + dz, y + dy, x + dx):
                    if levels[z + dz, y + dy, x + dx] == levels[z, y, x]:
                        step = distance[z + dz, y + dy, x + dx]
                        if step < UNRESOLVED and step + 1 < best:
                            best = step + 1
            if best < current:
                distance[z, y, x] = best
                changed[0] = 1


@ti.kernel
def _parents(levels: F32, heat: F32, seeds: I32, distance: I32, parent: I32,
             z0: ti.i32, y0: ti.i32, x0: ti.i32, z1: ti.i32, y1: ti.i32, x1: ti.i32):
    # The parent rule of gpu_watershed._directions, completed with the plateau rule of its
    # breadth-first rounds. Only voxels inside the core box get a parent; the halo keeps its labels.
    for z, y, x in levels:
        parent[z, y, x] = UNLABELED
        if z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1 and levels[z, y, x] > 0.0:
            if seeds[z, y, x] != 0:
                parent[z, y, x] = index_of(levels, z, y, x)
            else:
                best, best_heat, target = levels[z, y, x], -ti.math.inf, FLAT
                for dz, dy, dx in ti.static(NEIGHBOURS):
                    if inside(levels, z + dz, y + dy, x + dx):
                        value, height = levels[z + dz, y + dy, x + dx], heat[z + dz, y + dy, x + dx]
                        if value > best or (value == best == ti.math.inf and height > best_heat):
                            best, best_heat, target = value, height, index_of(levels, z + dz, y + dy, x + dx)
                if target == FLAT:  # plateau: the first neighbour one step closer to its exit
                    for dz, dy, dx in ti.static(NEIGHBOURS):
                        if target == FLAT and inside(levels, z + dz, y + dy, x + dx):
                            if (levels[z + dz, y + dy, x + dx] == levels[z, y, x]
                                    and distance[z + dz, y + dy, x + dx] == distance[z, y, x] - 1):
                                target = index_of(levels, z + dz, y + dy, x + dx)
                parent[z, y, x] = target


@ti.kernel
def _pull_labels(parent: I32, labels: I32, changed: LIST):
    for z, y, x in parent:
        up = parent[z, y, x]
        if up >= 0 and labels[z, y, x] == 0:
            uz, uy, ux = voxel_of(parent, up)
            if labels[uz, uy, ux] != 0:
                labels[z, y, x] = labels[uz, uy, ux]
                changed[0] = 1


@frees_gpu_memory_here
def _raise_levels_on_gpu(marker, ceiling):
    m, g = to_device(marker, np.float32), to_device(ceiling, np.float32)
    reconstruct_by_dilation_(m, g)
    return m.to_numpy().reshape(marker.shape)


@frees_gpu_memory_here
def _relax_distance_on_gpu(levels, distance):
    levels_device, distance_device = to_device(levels, np.float32), to_device(distance, np.int32)
    until_stable(lambda flag, _: _relax_plateau_distance(levels_device, distance_device, flag))
    return distance_device.to_numpy().reshape(distance.shape)


@frees_gpu_memory_here
def _label_on_gpu(levels, heat, seeds, distance, labels, core):
    levels_device, heat_device = to_device(levels, np.float32), to_device(heat, np.float32)
    seeds_device, distance_device = to_device(seeds, np.int32), to_device(distance, np.int32)
    labels_device = to_device(labels, np.int32)
    parent = ti.ndarray(ti.i32, levels.shape)
    _parents(levels_device, heat_device, seeds_device, distance_device, parent,
             core[0].start, core[1].start, core[2].start, core[0].stop, core[1].stop, core[2].stop)
    until_stable(lambda flag, _: _pull_labels(parent, labels_device, flag))
    return labels_device.to_numpy().reshape(labels.shape)


# ─────────────────────────── host-side phases ───────────────────────────

def _flood_ceiling(heat, seeds, background):
    """What gpu_watershed._shave_inputs writes as `mask`: the heat above the background, infinite at seeds."""
    ceiling = np.where(heat > background, heat, np.float32(0))
    ceiling[(seeds != 0) & (ceiling > 0)] = np.inf
    return ceiling


def _initial_levels(heat, seeds, background, slab=32):
    """What gpu_watershed._shave_inputs writes as `marker`: the ceiling at seeds, zero elsewhere."""
    levels = np.empty(heat.shape, np.float32)
    for start in range(0, heat.shape[0], slab):
        box = slice(start, start + slab)
        levels[box] = np.where(seeds[box] != 0, _flood_ceiling(heat[box], seeds[box], background), np.float32(0))
    return levels


def _initial_plateau_distance(levels, seeds):
    """UNRESOLVED on plateau voxels (reached, not a seed, no higher neighbour), 0 everywhere else."""
    plateau = (levels > 0) & (seeds == 0)
    for axis in range(3):
        low = tuple(slice(0, -1) if a == axis else slice(None) for a in range(3))
        high = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        plateau[low] &= ~(levels[high] > levels[low])
        plateau[high] &= ~(levels[low] > levels[high])
    return np.where(plateau, UNRESOLVED, 0).astype(np.int32)


def _reconstruct_patched(levels, ceiling_of, budget, phase):
    """Reconstruction by dilation of `levels` (in place) under ceiling_of(box), patch by patch."""
    def update(patch):
        marker = levels[patch.padded]
        if not marker.any():  # nothing to flood from
            return False
        budget.check_free(patch, LEVELS_BYTES)
        raised = _raise_levels_on_gpu(marker, ceiling_of(patch.padded))[patch.core_in_padded]
        changed = not np.array_equal(raised, levels[patch.core])
        levels[patch.core] = raised
        return changed

    run_to_fixed_point(levels.shape, budget, update, phase=phase,
                       raw_bytes_per_voxel=LEVELS_BYTES)


def _settle_plateau_distance(levels, distance, budget):
    def update(patch):
        if not distance[patch.core].any():  # no plateau voxels here
            return False
        budget.check_free(patch, DISTANCE_BYTES)
        settled = _relax_distance_on_gpu(levels[patch.padded], distance[patch.padded])[patch.core_in_padded]
        changed = not np.array_equal(settled, distance[patch.core])
        distance[patch.core] = settled
        return changed

    run_to_fixed_point(levels.shape, budget, update, phase="plateau distances",
                       raw_bytes_per_voxel=DISTANCE_BYTES)


def _propagate_labels(levels, heat, seeds, distance, labels, budget):
    def update(patch):
        waiting = (levels[patch.core] > 0) & (labels[patch.core] == 0)
        if not waiting.any():  # nothing to label here
            return False
        budget.check_free(patch, LABELS_BYTES)
        labelled = _label_on_gpu(levels[patch.padded], heat[patch.padded], seeds[patch.padded],
                                 distance[patch.padded], labels[patch.padded], patch.core_in_padded)
        labelled = labelled[patch.core_in_padded]
        changed = not np.array_equal(labelled, labels[patch.core])
        labels[patch.core] = labelled
        return changed

    run_to_fixed_point(levels.shape, budget, update, phase="labels",
                       raw_bytes_per_voxel=LABELS_BYTES)


def _watershed_patched(heat, seeds, background_threshold, budget):
    background = np.float32(background_threshold)
    levels = _initial_levels(heat, seeds, background)
    _reconstruct_patched(levels, lambda box: _flood_ceiling(heat[box], seeds[box], background),
                         budget, "flood levels")

    patches = plan_patches(heat.shape, budget.limit_voxels(LABELS_BYTES))
    unreached = sum(not (levels[patch.core] > 0).any() for patch in patches)
    log.info("seed scan: %d of %d patches are out of reach of every seed and stay background",
             unreached, len(patches))

    distance = _initial_plateau_distance(levels, seeds)
    _settle_plateau_distance(levels, distance, budget)
    labels = np.where(np.isinf(levels), seeds, 0).astype(np.int32)
    _propagate_labels(levels, heat, seeds, distance, labels, budget)
    if ((levels > 0) & (labels == 0)).any():
        raise RuntimeError("unresolved flat voxels remain; the shave guarantee was violated")
    return labels


def _unpatched_or_patched(shape, budget, raw_bytes_per_voxel, phase, unpatched, patched):
    """`unpatched()` when one patch covers the volume, else `patched()`. The GPU running out of
    memory on the single patch halves the budget once (see PatchBudget) and tries again."""
    while True:
        plan = plan_patches(shape, budget.limit_voxels(raw_bytes_per_voxel))
        if len(plan) > 1:
            return patched()
        def attempt():
            budget.check_free(plan[0], raw_bytes_per_voxel)
            return unpatched()

        result, oom = call_or_oom(attempt)
        if oom is None:
            budget.recovering = False
            return result
        budget.halve_after_oom(plan[0], phase, raw_bytes_per_voxel)


def _as_volume(array, dtype):
    array = np.asarray(array, dtype)
    return array[None] if array.ndim == 2 else array


# ───────────────────────────── public API ─────────────────────────────

def h_dome(image, h, *, budget=None):
    """h-dome transform image - R(image - h), numpy float32 in and out; see gpu_morphology.h_dome."""
    volume = _as_volume(image, np.float32)
    budget = budget or watershed_budget(volume.size)

    def patched():
        marker = np.minimum(volume - np.float32(h), volume)
        _reconstruct_patched(marker, lambda box: volume[box], budget, "h-dome")
        return volume - marker

    result = _unpatched_or_patched(volume.shape, budget, LEVELS_BYTES, "h-dome",
                                   lambda: h_dome_whole(volume, h), patched)
    return result.reshape(np.shape(image))


def seeded_watershed(heat, seeds, background_threshold, *, budget=None):
    """numpy in (heat float, seeds int), numpy int32 labels out; see gpu_watershed.seeded_watershed."""
    heat_volume, seeds_volume = _as_volume(heat, np.float32), _as_volume(seeds, np.int32)
    budget = budget or watershed_budget(heat_volume.size)
    labels = _unpatched_or_patched(
        heat_volume.shape, budget, LABELS_BYTES, "watershed",
        lambda: seeded_watershed_whole(heat_volume, seeds_volume, background_threshold),
        lambda: _watershed_patched(heat_volume, seeds_volume, background_threshold, budget))
    return labels.reshape(np.shape(heat))
