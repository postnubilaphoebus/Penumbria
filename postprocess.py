from scipy.ndimage import label, gaussian_filter
import numpy as np
from matching import matching
import numba as nb
from scipy.ndimage import sobel
from gpu_patched import h_dome, seeded_watershed  # as gpu_morphology / gpu_watershed, within a GPU memory budget
from skimage.segmentation import watershed
import math
import warnings
import psutil
from pathlib import Path
from numcodecs import Blosc
from volume_io import (
    ChunkedVolume,
    build_sampling_index,
    create_ome_zarr,
    create_zarr_array,
    iter_chunk_slices,
    update_group_attributes,
    write_zarr_region,
)

# Below this many voxels a whole image is flooded by skimage's watershed on the CPU instead of the GPU one,
# which has a fixed cost of about 0.1 s per call. On the laptop GPU it was tested on (RTX 4050), skimage is
# faster below roughly 0.7M voxels, the two tie up to about 1M, and the GPU wins above that.
SKIMAGE_WATERSHED_BELOW_VOXELS = 2_500_000


def gradient_symmetry_voting(image, r=5):
    gx = sobel(image, axis=0)
    gy = sobel(image, axis=1)
    gz = sobel(image, axis=2)
    return vote_map_3d(gx, gy, gz, r)


@nb.njit(parallel=True)
def vote_map_3d(gx, gy, gz, r):
    shape = gx.shape
    out = np.zeros(shape, dtype=np.float32)
    eps = 1e-8

    for x in nb.prange(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # gradient vector
                gxv = gx[x, y, z]
                gyv = gy[x, y, z]
                gzv = gz[x, y, z]

                mag = np.sqrt(gxv**2 + gyv**2 + gzv**2)
                if mag > eps:
                    # unit gradient
                    ux = gxv / mag
                    uy = gyv / mag
                    uz = gzv / mag

                    # positive vote
                    cx = int(round(x + ux * r))
                    cy = int(round(y + uy * r))
                    cz = int(round(z + uz * r))
                    if 0 <= cx < shape[0] and 0 <= cy < shape[1] and 0 <= cz < shape[2]:
                        out[cx, cy, cz] += 1

                    # negative vote
                    cx = int(round(x - ux * r))
                    cy = int(round(y - uy * r))
                    cz = int(round(z - uz * r))
                    if 0 <= cx < shape[0] and 0 <= cy < shape[1] and 0 <= cz < shape[2]:
                        out[cx, cy, cz] -= 1

    return out

def muti_scale_symmetry(img, background_threshold = 0.05):
    background_mask = img > background_threshold
    res_map = []
    for i in range(7, 14, 2):
        res = gradient_symmetry_voting(img, i)
        res = gaussian_filter(res, sigma=0.5)
        res = (res - res.min()) / (res.max() - res.min())
        res_map.append(res)
    res_map = np.array(res_map)
    res_map = np.mean(res_map, axis=0)
    res_map[~background_mask] = 0
    return res_map
    
def has_nonzero_elements(lst):
    if all(isinstance(el, list) for el in lst):
        return any(any(sublist) for sublist in lst)
    else:
        return any(lst)
    
def twice_smooth_and_threshold(img):
    gfilt = gaussian_filter(img, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    gfilt = gaussian_filter(gfilt, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    return gfilt

def watershed_inference(prediction,
                        padding,
                        minimum_cell_size = 9,
                        h = 0.1, 
                        cell_confidence_minimum = 0.5, 
                        background_threshold = 0.2,
                        gaussian_smoothing = True,
                        simple_thresholding = False,
                        low_confidence_merging = False,
                        sym = False,
                        skimage_below_voxels=SKIMAGE_WATERSHED_BELOW_VOXELS,
                        _smoothed_prediction=None,
                        _normalization_bounds=None):
    
    if padding is not None and has_nonzero_elements(padding):
        shape_trimming_by_extra_width = True
    else:
        shape_trimming_by_extra_width = False
    prediction = np.asarray(prediction, dtype=np.float32)
    hdome_image = prediction.copy()

    if gaussian_smoothing:
        hdome_image = (gaussian_filter(hdome_image, sigma=1) if _smoothed_prediction is None
                       else _smoothed_prediction)
        low, high = ((hdome_image.min(), hdome_image.max()) if _normalization_bounds is None
                     else _normalization_bounds)
        if high != low:
            hdome_image = (hdome_image - low) / (high - low + 1e-8)
        else:
            hdome_image = np.zeros_like(hdome_image)
    if hdome_image.sum() == 0:
        if shape_trimming_by_extra_width:
            if prediction.ndim == 3:
                return np.zeros_like(prediction)[padding[0][0]:prediction.shape[0]-padding[0][1], 
                                                 padding[1][0]:prediction.shape[1]-padding[1][1], 
                                                 padding[2][0]:prediction.shape[2]-padding[2][1]]
            elif prediction.ndim == 2:
                return np.zeros_like(prediction)[padding[0][0]:prediction.shape[0]-padding[0][1], 
                                                 padding[1][0]:prediction.shape[1]-padding[1][1]]
            else:
                raise ValueError(f"incorrect image shape{prediction.shape}")
        else:
            return np.zeros_like(prediction)
        
    if not simple_thresholding:
        # hdome transform
        if sym:
            hdome_image = muti_scale_symmetry(hdome_image)
        h_maxima_binary = (h_dome(hdome_image, h) > 0) & (prediction > h)
        labeled_array, _ = label(h_maxima_binary)
    else:
        h_maxima_binary = hdome_image > h
        labeled_array, _ = label(h_maxima_binary)


    # The GPU watershed is equivalent to skimage's watershed(-prediction, markers, mask=...), which is
    # the faster one for a small whole image (tiles of a large image pass 0 and always use the GPU).
    del hdome_image, h_maxima_binary
    if prediction.size < skimage_below_voxels:
        wts = watershed(-prediction, labeled_array, mask=prediction > background_threshold)
    else:
        wts = seeded_watershed(prediction, labeled_array, background_threshold)
    del labeled_array

    # Exactly the original strict count/max-confidence tests, without several
    # full-cell int64 coordinate arrays when a single cell occupies most of RAM.
    return _filter_watershed_labels(wts, prediction, minimum_cell_size, cell_confidence_minimum)


def _filter_watershed_labels(labels, prediction, minimum_cell_size, confidence):
    maximum = int(labels.max())
    counts = np.zeros(maximum + 1, dtype=np.int64)
    maxima = np.full(maximum + 1, -np.inf, dtype=np.float32)
    chunks = tuple(min(size, 64 if axis == 0 else 256) for axis, size in enumerate(labels.shape))
    for region in iter_chunk_slices(labels.shape, chunks):
        ids = labels[region].ravel()
        counts += np.bincount(ids, minlength=maximum + 1)
        np.maximum.at(maxima, ids, prediction[region].ravel())
    keep = (counts > minimum_cell_size) & (maxima > confidence)
    keep[0] = False
    mapping = np.zeros(maximum + 1, dtype=labels.dtype)
    mapping[keep] = np.arange(1, np.count_nonzero(keep) + 1, dtype=labels.dtype)
    for region in iter_chunk_slices(labels.shape, chunks):
        labels[region] = mapping[labels[region]]
    return labels


def estimate_watershed_memory_bytes(shape, *, simple_thresholding=False):
    """Conservative working-set estimate, not just float32 input size.

    Covers padded float64 skimage input, markers/masks/labels, queue headroom,
    marker preparation and filtering. Reconstruction gets extra headroom.
    It is a heuristic, not a hard bound on the data-dependent watershed heap.
    """
    shape = tuple(int(size) for size in shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("Watershed requires a non-empty 3-D volume")
    return math.prod(size + 2 for size in shape) * (80 if simple_thresholding else 96) + 256 * 1024**2


def watershed_memory_budget_bytes(memory_budget_bytes=None):
    """Leave at least 4 GiB free and use at most 80% of currently available RAM."""
    memory = psutil.virtual_memory()
    budget = max(0, min(int(memory.total * 0.75), int(memory.available * 0.8),
                        int(memory.available) - 4 * 1024**3))
    if memory_budget_bytes is not None:
        if memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        budget = min(budget, int(memory_budget_bytes))
    return budget


def watershed_inference_auto(
    prediction, output_path, *, minimum_cell_size=9, h=0.1,
    cell_confidence_minimum=0.5, background_threshold=0.2,
    gaussian_smoothing=True, simple_thresholding=False,
    memory_budget_bytes=None, tile_shape=(128, 256, 256), halo=32,
):
    """Prefer ONE global GPU watershed; tile only above the RAM budget.

    Disk storage chunks never define watershed boundaries on the global path.
    Both paths return a uint32 OME-Zarr volume for the existing ImageJ export.
    """
    shape = tuple(prediction.shape)
    estimate = estimate_watershed_memory_bytes(shape, simple_thresholding=simple_thresholding)
    budget = watershed_memory_budget_bytes(memory_budget_bytes)
    settings = dict(minimum_cell_size=minimum_cell_size, h=h,
                    cell_confidence_minimum=cell_confidence_minimum,
                    background_threshold=background_threshold,
                    gaussian_smoothing=gaussian_smoothing, simple_thresholding=simple_thresholding)
    metadata = {"penumbria_watershed_estimated_memory_bytes": estimate,
                "penumbria_watershed_memory_budget_bytes": budget}
    if estimate <= budget:
        print(f"Global GPU watershed: {shape}, estimated {estimate / 1024**3:.2f} GiB; "
              f"budget {budget / 1024**3:.2f} GiB. No watershed tiles.")
        if isinstance(prediction, ChunkedVolume):
            # One allocation, bounded decoding buffers; avoid parallel Zarr
            # whole-volume decoding creating additional full-sized copies.
            image = np.empty(shape, dtype=np.float32)
            for region in iter_chunk_slices(shape, prediction.chunks):
                image[region] = prediction.read_region(region)
        else:
            image = np.asarray(prediction, dtype=np.float32)
        labels = watershed_inference(image, padding=None, **settings)
        del image
        count = int(labels.max())
        group, output = create_ome_zarr(output_path, shape, np.uint32, is_label=True, overwrite=True)
        for region in iter_chunk_slices(shape, output.chunks):
            write_zarr_region(output, region, labels[region].astype(np.uint32, copy=False))
        update_group_attributes(group, {**metadata, "penumbria_complete": True,
                                       "penumbria_num_labels": count,
                                       "penumbria_watershed_method": "global_gpu"})
        return ChunkedVolume(output_path), count

    warnings.warn(
        f"Global watershed estimate {estimate / 1024**3:.2f} GiB exceeds the available "
        f"budget {budget / 1024**3:.2f} GiB. Using approximate tiled GPU watershed "
        "with strict overlap-continuity checks; this is not equivalent to global flooding.",
        RuntimeWarning, stacklevel=2,
    )
    tile_shape = tuple(min(int(size), int(tile)) for size, tile in zip(shape, tile_shape))
    if len(tile_shape) != 3 or any(tile <= 0 for tile in tile_shape) or halo < 1:
        raise ValueError("tile_shape must have three positive dimensions and halo must be positive")
    # The fallback must also respect the current budget (including its halo).
    while True:
        expanded = tuple(min(size, tile + 2 * halo) for size, tile in zip(shape, tile_shape))
        # Extra copies, overlap pair sorting, union-find and disk write buffers.
        tile_estimate = estimate_watershed_memory_bytes(expanded, simple_thresholding=simple_thresholding) * 2
        if tile_estimate <= budget:
            break
        axis = max(range(3), key=lambda index: tile_shape[index])
        if tile_shape[axis] == 1:
            raise MemoryError("Not enough available RAM even for a watershed tile and its halo; close other programs.")
        smaller = list(tile_shape)
        smaller[axis] = max(1, smaller[axis] // 2)
        tile_shape = tuple(smaller)
    result, count = watershed_inference_chunked(
        prediction, output_path, tile_shape=tile_shape, halo=halo,
        memory_budget_bytes=budget, **settings,
    )
    group, _ = create_ome_zarr(result.path, shape, np.uint32, is_label=True)
    update_group_attributes(group, metadata)
    return result, count


class _UnionFind:
    def __init__(self):
        self.parent = [0]

    def extend_to(self, maximum):
        self.parent.extend(range(len(self.parent), int(maximum) + 1))

    def find(self, value):
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _merge_consistent_overlap_pairs(union_find, previous, current, processed):
    """Accept only equivalent overlap partitions, allowing arbitrary label IDs.

    Mutual-best matching alone can silently discard split/merge conflicts.
    Zero is meaningful only where a previous tile actually wrote a core.
    """
    mismatch = processed & ((previous > 0) != (current > 0))
    if np.any(mismatch):
        raise RuntimeError(
            "Tiled watershed continuity check failed: neighboring tiles disagree on labeled "
            "foreground. A marker may lie outside the halo or differ between tiles. "
            "No completed segmentation was saved. Free RAM for global watershed, "
            "or use a larger tile/halo with sufficient memory."
        )
    mask = processed & (previous > 0) & (current > 0)
    if not np.any(mask):
        return
    pairs = np.unique(np.stack((previous[mask], current[mask]), axis=1), axis=0)
    canonical_pairs = {(union_find.find(left), union_find.find(right)) for left, right in pairs}
    if (len({left for left, _ in canonical_pairs}) != len(canonical_pairs)
            or len({right for _, right in canonical_pairs}) != len(canonical_pairs)):
        raise RuntimeError(
            "Tiled watershed continuity check failed: overlapping labels have an ambiguous "
            "split/merge or shifted boundary. No completed segmentation was saved. "
            "Use global watershed or increase the tile/halo with sufficient memory."
        )
    for previous_id, current_id in canonical_pairs:
        union_find.union(previous_id, current_id)


def _smoothed_region(prediction, region):
    """sigma=1 with the full four-sigma support, including at storage edges."""
    expanded = tuple(slice(max(0, sl.start - 4), min(size, sl.stop + 4))
                     for sl, size in zip(region, prediction.shape))
    values = (prediction.read_region(expanded) if isinstance(prediction, ChunkedVolume)
              else prediction[expanded])
    smooth = gaussian_filter(np.asarray(values, dtype=np.float32), sigma=1)
    crop = tuple(slice(sl.start - extra.start, sl.stop - extra.start)
                 for sl, extra in zip(region, expanded))
    return smooth[crop].copy()


def _global_smoothed_bounds(prediction, chunks):
    low, high = np.float32(np.inf), np.float32(-np.inf)
    for region in iter_chunk_slices(prediction.shape, chunks):
        values = _smoothed_region(prediction, region)
        low, high = min(low, values.min()), max(high, values.max())
    return low, high


def watershed_inference_chunked(
    prediction,
    output_path,
    *,
    minimum_cell_size=9,
    h=0.1,
    cell_confidence_minimum=0.5,
    background_threshold=0.2,
    gaussian_smoothing=True,
    simple_thresholding=False,
    tile_shape=(128, 256, 256),
    halo=32,
    memory_budget_bytes=None,
):
    """Run overlapping GPU h-dome/watersheds and merge them on disk.

    Approximate fallback, NOT equivalent to global flooding. Overlaps must
    agree exactly up to label IDs; inconsistent coverage or split/merge
    conflicts raise before marking the output complete. This detects local
    inconsistencies, not all possible finite-halo errors. Prefer the auto
    dispatcher, which uses global watershed whenever its RAM estimate fits.
    """
    if not isinstance(prediction, ChunkedVolume):
        prediction = np.asarray(prediction)
    shape = tuple(prediction.shape)
    if len(shape) != 3:
        raise ValueError("Chunked watershed supports 3-D predictions only")
    if len(tile_shape) != 3 or any(int(value) <= 0 for value in tile_shape) or int(halo) < 1:
        raise ValueError("tile_shape must have three positive dimensions and halo must be positive")
    tile_shape = tuple(min(shape[axis], int(tile_shape[axis])) for axis in range(3))
    budget = watershed_memory_budget_bytes(memory_budget_bytes)
    expanded_shape = tuple(min(size, tile + 2 * halo) for size, tile in zip(shape, tile_shape))
    tile_memory = 2 * estimate_watershed_memory_bytes(expanded_shape, simple_thresholding=simple_thresholding)
    if tile_memory > budget:
        raise MemoryError("Watershed tile and halo exceed the available RAM budget; reduce the tile size.")
    group, output = create_ome_zarr(
        output_path, shape, np.uint32, chunks=tile_shape,
        name=f"{Path(output_path).stem} labels", is_label=True, overwrite=True,
    )
    update_group_attributes(group, {"penumbria_complete": False,
                                   "penumbria_watershed_method": "tiled_gpu",
                                   "penumbria_watershed_overlap_check": "pending"})
    raw = create_zarr_array(
        group, "_raw", shape=shape, chunks=tile_shape, dtype=np.uint64, fill_value=0,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE), overwrite=True,
    )
    union_find = _UnionFind()
    label_offset = 0
    normalization_bounds = (_global_smoothed_bounds(prediction, tile_shape)
                            if gaussian_smoothing else None)

    for core in iter_chunk_slices(shape, tile_shape):
        core_start = np.asarray([sl.start for sl in core], dtype=np.int64)
        core_end = np.asarray([sl.stop for sl in core], dtype=np.int64)
        expanded_start = np.maximum(0, core_start - int(halo))
        expanded_end = np.minimum(np.asarray(shape), core_end + int(halo))
        expanded_region = tuple(slice(int(expanded_start[a]), int(expanded_end[a])) for a in range(3))
        if isinstance(prediction, ChunkedVolume):
            local_prediction = prediction.read_region(expanded_region)
        else:
            local_prediction = np.asarray(prediction[expanded_region])

        local_labels = watershed_inference(
            local_prediction,
            padding=None,
            minimum_cell_size=0,
            h=h,
            cell_confidence_minimum=-np.inf,
            background_threshold=background_threshold,
            gaussian_smoothing=gaussian_smoothing,
            simple_thresholding=simple_thresholding,
            low_confidence_merging=False,
            sym=False,
            skimage_below_voxels=0,
            _smoothed_prediction=(_smoothed_region(prediction, expanded_region)
                                  if gaussian_smoothing else None),
            _normalization_bounds=normalization_bounds,
        ).astype(np.uint64, copy=False)
        local_max = int(local_labels.max())
        # Python union-find integers/list pointers and the later count/max/root
        # arrays grow with ALL tile-local IDs, including discarded halo labels.
        # Do not let a high-density enormous volume exhaust RAM in bookkeeping.
        if tile_memory + 128 * (label_offset + local_max + 1) > budget:
            raise MemoryError(
                "Tiled watershed label bookkeeping exceeds the RAM budget. "
                "No completed segmentation was saved; use a machine with more RAM."
            )
        union_find.extend_to(label_offset + local_max)
        current_global = np.where(local_labels > 0, local_labels + label_offset, 0).astype(np.uint64)

        previous_global = np.asarray(raw[expanded_region], dtype=np.uint64)
        z = np.arange(expanded_start[0], expanded_end[0])[:, None, None]
        y = np.arange(expanded_start[1], expanded_end[1])[None, :, None]
        x = np.arange(expanded_start[2], expanded_end[2])[None, None, :]
        # iter_chunk_slices visits Z/Y/X cores lexicographically. Distinguish
        # already-written background zeros from not-yet-visited zeros.
        processed = ((z < core_start[0])
                     | ((z < core_end[0]) & (y < core_start[1]))
                     | ((z < core_end[0]) & (y < core_end[1]) & (x < core_start[2])))
        _merge_consistent_overlap_pairs(union_find, previous_global, current_global, processed)

        local_core = tuple(
            slice(int(core_start[a] - expanded_start[a]), int(core_end[a] - expanded_start[a]))
            for a in range(3)
        )
        write_zarr_region(raw, core, current_global[local_core])
        label_offset += local_max

    counts = np.zeros(label_offset + 1, dtype=np.uint64)
    maxima = np.full(label_offset + 1, -np.inf, dtype=np.float32)
    for region in iter_chunk_slices(shape, raw.chunks):
        raw_chunk = np.asarray(raw[region], dtype=np.int64)
        if isinstance(prediction, ChunkedVolume):
            prediction_chunk = np.asarray(prediction.read_region(region), dtype=np.float32)
        else:
            prediction_chunk = np.asarray(prediction[region], dtype=np.float32)
        ids, local_counts = np.unique(raw_chunk[raw_chunk > 0], return_counts=True)
        counts[ids] += local_counts.astype(np.uint64)
        np.maximum.at(maxima, raw_chunk.ravel(), prediction_chunk.ravel())

    root_counts = np.zeros_like(counts)
    root_maxima = np.full_like(maxima, -np.inf)
    active_ids = np.flatnonzero(counts)
    for label_id in active_ids:
        root = union_find.find(int(label_id))
        root_counts[root] += counts[label_id]
        root_maxima[root] = max(root_maxima[root], maxima[label_id])

    mapping = np.zeros(label_offset + 1, dtype=np.uint32)
    next_label = 1
    for root in np.flatnonzero(root_counts):
        if root_counts[root] > minimum_cell_size and root_maxima[root] > cell_confidence_minimum:
            mapping[root] = next_label
            next_label += 1
    for label_id in active_ids:
        mapping[label_id] = mapping[union_find.find(int(label_id))]

    for region in iter_chunk_slices(shape, output.chunks):
        write_zarr_region(output, region, mapping[np.asarray(raw[region], dtype=np.int64)])
    del group["_raw"]
    update_group_attributes(group, {
        "penumbria_complete": True,
        "penumbria_num_labels": int(next_label - 1),
        "penumbria_watershed_tile_shape": tile_shape,
        "penumbria_watershed_halo": int(halo),
        "penumbria_watershed_overlap_check": "passed",
    })
    return ChunkedVolume(output_path), int(next_label - 1)

def prepare_tuning_patches(predictions, integer_labels, images, training_image_shape, max_patches_per_image=8):
    """Validation regions for Optuna: prediction and labels cut from the same voxels, always inside the volume.

    A validation image is tuned whole, as long as that is no more expensive than the windows would be.
    Only a larger image gets windows, centred on cells where possible. They are shifted to stay inside the
    volume: padding would put mirrored copies of the cells into the prediction and nothing into the
    labels, and the objective counts those mirrored cells as false positives.
    """
    tuning_predictions, tuning_labels = [], []
    tuning_shape = tuple(max(192, int(value) * 2) for value in training_image_shape)
    for prediction, integer_label, image in zip(predictions, integer_labels, images):
        window = tuple(min(prediction.shape[axis], tuning_shape[axis]) for axis in range(3))
        if math.prod(prediction.shape) <= max_patches_per_image * math.prod(window):
            starts = [(0, 0, 0)]
            window = tuple(prediction.shape)
        else:
            centres = build_sampling_index(integer_label, image, training_image_shape).object_centres
            if len(centres) == 0:
                centres = np.asarray([np.asarray(prediction.shape) // 2])
            if len(centres) > max_patches_per_image:
                centres = centres[np.linspace(0, len(centres) - 1, max_patches_per_image, dtype=int)]
            starts = list(dict.fromkeys(  # identical windows only once
                tuple(int(np.clip(centre[axis] - window[axis] // 2, 0, prediction.shape[axis] - window[axis]))
                      for axis in range(3))
                for centre in centres
            ))
        for start in starts:
            region = tuple(slice(start[axis], start[axis] + window[axis]) for axis in range(3))
            tuning_predictions.append(prediction.read_region(region))
            tuning_labels.append(integer_label.read_region(region))
    return tuning_predictions, tuning_labels


def objective(trial, img, target, padding_list_val, data_dimensionality):
    gaussian_smoothing = trial.suggest_categorical('gaussian_smoothing', [True, False])
    h = trial.suggest_float('h', 0.15, 0.5, step=0.01)
    bg = trial.suggest_float('bg', 0.05, 0.25, step=0.01)
    c = trial.suggest_float('c', 0.2, 0.75, step=0.01)
    simple_thresholding = trial.suggest_categorical('simple_thresholding', [True, False])
    sym = False
    lcm = False
    sum_acc = 0
    map_vals = np.arange(0.1, 1.0, 0.1) if data_dimensionality == 3 else np.arange(0.5, 1.0, 0.05)
    for im, ta, pa in zip(img, target, padding_list_val):
        try:
            pred = watershed_inference(
                prediction=im,
                padding = pa,
                minimum_cell_size=9,
                h=h,
                cell_confidence_minimum=c,
                background_threshold=bg,
                gaussian_smoothing=gaussian_smoothing,
                simple_thresholding=simple_thresholding,
                low_confidence_merging=lcm,
                sym=sym
            )
            mean_acc = 0
            idx = 0
            for val in map_vals:
                stats_dict = matching(ta, pred, val)
                mean_acc += stats_dict.accuracy
                idx += 1
            sum_acc += (mean_acc / idx)
        except:
            sum_acc += 0
        
    return sum_acc / len(img)

if __name__ == "__main__":
    pass

