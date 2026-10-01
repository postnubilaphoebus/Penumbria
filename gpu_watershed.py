"""Seeded watershed on the GPU (Taichi/CUDA), 6-connectivity.

Every foreground voxel climbs the heatmap to a seed. Seed voxels count as
infinitely high, as in skimage, whose markers enter the queue at -inf: a voxel
touching a seed joins it, and seed heights never limit a path.
1. shave       reconstruction by dilation cuts every unseeded bump down to
               its saddle, so the only maxima left are seeds;
2. directions  each voxel points at its highest neighbour;
3. flats       voxels without a higher neighbour point, breadth-first, at an
               equal neighbour closer to the flat's exit;
4. jumping     parent <- parent[parent] until every voxel points at its seed.

Equivalent to skimage.segmentation.watershed(-heat, seeds, mask=heat >
background_threshold), whose default connectivity is also 6: seeds outside the
mask are dropped, foreground connected to no seed stays 0.

Usage (CUDA starts on first call; 2-D inputs are accepted):
    labels = seeded_watershed(heat, seeds, background_threshold=0.1)
"""
import numpy as np
import taichi as ti

from gpu_morphology import (F32, I32, LIST, NEIGHBOURS, check_volume, frees_gpu_memory_here,
                            index_of, inside, reconstruct_by_dilation_, to_device, until_stable,
                            voxel_of)

UNLABELED, FLAT, UNRESOLVED = -1, -2, 2**31 - 1


@ti.kernel
def _shave_inputs(heat: F32, seeds: I32, background: ti.f32, marker: F32, mask: F32):
    for z, y, x in heat:
        value = heat[z, y, x] if heat[z, y, x] > background else 0.0
        if value > 0.0 and seeds[z, y, x] != 0:
            value = ti.math.inf
        mask[z, y, x] = value
        marker[z, y, x] = value if seeds[z, y, x] != 0 else 0.0


@ti.kernel
def _directions(s: F32, heat: F32, seeds: I32, parent: I32, dist: I32, flats: LIST, n_flats: LIST):
    # s == 0 exactly where no path within the foreground reaches a seed. A voxel
    # touching several seeds (all infinitely high in s) joins the highest in heat.
    for z, y, x in s:
        dist[z, y, x] = 0
        if s[z, y, x] <= 0.0:
            parent[z, y, x] = UNLABELED
        elif seeds[z, y, x] != 0:
            parent[z, y, x] = index_of(s, z, y, x)
        else:
            best, best_heat, target = s[z, y, x], -ti.math.inf, FLAT
            for dz, dy, dx in ti.static(NEIGHBOURS):
                if inside(s, z + dz, y + dy, x + dx):
                    value, height = s[z + dz, y + dy, x + dx], heat[z + dz, y + dy, x + dx]
                    if value > best or (value == best == ti.math.inf and height > best_heat):
                        best, best_heat, target = value, height, index_of(s, z + dz, y + dy, x + dx)
            parent[z, y, x] = target
            if target == FLAT:
                dist[z, y, x] = UNRESOLVED
                flats[ti.atomic_add(n_flats[0], 1)] = index_of(s, z, y, x)


@ti.kernel
def _flat_round(s: F32, parent: I32, dist: I32, flats: LIST, n: ti.i32, round_: ti.i32, changed: LIST):
    # Only neighbours resolved in earlier rounds count, so in-place updates
    # stay an exact breadth-first search; the first such neighbour wins.
    for i in range(n):
        z, y, x = voxel_of(s, flats[i])
        for dz, dy, dx in ti.static(NEIGHBOURS):
            if dist[z, y, x] == UNRESOLVED and inside(s, z + dz, y + dy, x + dx):
                if s[z + dz, y + dy, x + dx] == s[z, y, x] and dist[z + dz, y + dy, x + dx] < round_:
                    parent[z, y, x] = index_of(s, z + dz, y + dy, x + dx)
                    dist[z, y, x] = round_
                    changed[0] = 1


@ti.kernel
def _count_unresolved(dist: I32, flats: LIST, n: ti.i32, count: LIST):
    for i in range(n):
        z, y, x = voxel_of(dist, flats[i])
        if dist[z, y, x] == UNRESOLVED:
            ti.atomic_add(count[0], 1)


@ti.kernel
def _jump(parent: I32, changed: LIST):
    # In place is safe: whatever value is read is still an ancestor.
    for z, y, x in parent:
        up = parent[z, y, x]
        if up >= 0:
            uz, uy, ux = voxel_of(parent, up)
            if parent[uz, uy, ux] != up:
                parent[z, y, x] = parent[uz, uy, ux]
                changed[0] = 1


@ti.kernel
def _roots_to_labels(parent: I32, seeds: I32):
    for z, y, x in parent:
        root = parent[z, y, x]
        label = 0
        if root >= 0:
            rz, ry, rx = voxel_of(parent, root)
            label = seeds[rz, ry, rx]
        parent[z, y, x] = label


def seeded_watershed_gpu(heat, seeds, background_threshold, *, return_stats=False, check_every=8):
    """Device arrays in (heat f32, seeds i32 with 0 = no seed), i32 labels out."""
    check_volume(heat, seeds)
    shape = heat.shape
    s, mask = ti.ndarray(ti.f32, shape), ti.ndarray(ti.f32, shape)
    _shave_inputs(heat, seeds, float(background_threshold), s, mask)
    shave_passes = reconstruct_by_dilation_(s, mask, check_every=check_every)
    del mask

    parent, dist = ti.ndarray(ti.i32, shape), ti.ndarray(ti.i32, shape)
    flats, count = ti.ndarray(ti.i32, shape[0] * shape[1] * shape[2]), ti.ndarray(ti.i32, 1)
    _directions(s, heat, seeds, parent, dist, flats, count)
    n_flats = int(count.to_numpy()[0])
    flat_rounds = 0
    if n_flats:
        flat_rounds = until_stable(
            lambda flag, r: _flat_round(s, parent, dist, flats, n_flats, r, flag), check_every)
        count.fill(0)
        _count_unresolved(dist, flats, n_flats, count)
        if count.to_numpy()[0]:
            raise RuntimeError("unresolved flat voxels remain; the shave guarantee was violated")
    del s, dist, flats

    jump_rounds = until_stable(lambda flag, _: _jump(parent, flag), check_every=1)
    _roots_to_labels(parent, seeds)
    if not return_stats:
        return parent
    return parent, dict(shave_passes=shave_passes, flat_voxels=n_flats,
                        flat_rounds=flat_rounds, jump_rounds=jump_rounds)


@frees_gpu_memory_here
def seeded_watershed(heat, seeds, background_threshold, **kwargs):
    """numpy in (heat float, seeds int), numpy int32 labels out."""
    result = seeded_watershed_gpu(to_device(heat, np.float32), to_device(seeds, np.int32),
                                  background_threshold, **kwargs)
    labels, stats = result if kwargs.get("return_stats") else (result, None)
    labels = labels.to_numpy().reshape(np.shape(heat))
    return (labels, stats) if stats is not None else labels
