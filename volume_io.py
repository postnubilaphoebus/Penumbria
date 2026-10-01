"""Out-of-core 3-D volume I/O used by training and inference.

TIFF files are converted once to a compressed, chunked OME-Zarr cache.  The
cache is deliberately a level-0 NGFF image: Fiji/ImageJ can inspect it with an
OME-Zarr/NGFF reader, while Penumbria can address small regions without ever
materialising the complete volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable, Iterator, Sequence

import numpy as np
from numcodecs import Blosc
import tifffile
import zarr
from scipy.ndimage import gaussian_filter, map_coordinates


DEFAULT_STORAGE_CHUNK_BYTES = 16 * 1024**2
DEFAULT_WORKING_MEMORY_BYTES = 4 * 1024**3
SUPPORTED_TIFF_SUFFIXES = {".tif", ".tiff"}
ZARR_MAJOR_VERSION = int(zarr.__version__.split(".", 1)[0])


def coerce_resizing_factor(value: object, max_denominator: int = 1_000_000) -> Fraction:
    """Return a positive, human-scale rational without binary-float noise."""
    if isinstance(value, Fraction):
        factor = value
    else:
        factor = Fraction(str(value).strip().strip("'\"")).limit_denominator(max_denominator)
    if factor <= 0:
        raise ValueError(f"Resizing factors must be positive, got {value!r}")
    return factor


def coerce_resizing_factors(values: Sequence[object]) -> tuple[Fraction, ...]:
    """Coerce scale values such as 0.333... to faithful fractions such as 1/3."""
    return tuple(coerce_resizing_factor(value) for value in values)


def parse_resizing_factors(text: str) -> tuple[Fraction, ...]:
    """Parse legacy decimal lists and new fractional resizing-factor lists."""
    values = [value.strip() for value in text.strip().strip("[]").split(",") if value.strip()]
    return coerce_resizing_factors(values)


def select_inference_resizing_factors(
    training_factors: Sequence[object], override: Sequence[object] | None
) -> tuple[Fraction, ...]:
    """Use an explicit inference override, otherwise inherit training geometry."""
    return coerce_resizing_factors(training_factors if override is None else override)


def _round_positive_fraction(value: Fraction) -> int:
    """Round a positive rational to the nearest voxel, with exact half-up ties."""
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def scaled_shape(
    shape: Sequence[int], factors: Sequence[object], *, inverse: bool = False
) -> tuple[int, ...]:
    """Scale a voxel shape exactly; round only the unavoidable final voxel counts."""
    rational_factors = coerce_resizing_factors(factors)
    if len(shape) != len(rational_factors):
        raise ValueError(f"Shape has {len(shape)} axes but {len(rational_factors)} factors were supplied")
    scaled = (
        Fraction(int(size), 1) / factor if inverse else Fraction(int(size), 1) * factor
        for size, factor in zip(shape, rational_factors)
    )
    return tuple(max(1, _round_positive_fraction(value)) for value in scaled)


def _open_group(path: os.PathLike[str] | str, mode: str) -> zarr.Group:
    """Open a Zarr-v2 group with either the Zarr 2 or Zarr 3 Python API."""
    if ZARR_MAJOR_VERSION >= 3:
        return zarr.open_group(str(path), mode=mode, zarr_format=2)
    return zarr.open_group(str(path), mode=mode)


def create_zarr_array(group: zarr.Group, name: str, **kwargs) -> zarr.Array:
    """Create an array through the Zarr 2 or Zarr 3 synchronous group API."""
    if ZARR_MAJOR_VERSION >= 3:
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def _retry_windows_file_lock(operation, *, attempts: int = 8):
    """Retry transient Windows failures replacing Zarr keys held by scanners."""
    delay = 0.05
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def update_group_attributes(group: zarr.Group, values: dict[str, object]) -> None:
    """Persist several attributes in one metadata write under Zarr 2 and 3."""
    merged = dict(group.attrs)
    merged.update(values)
    if ZARR_MAJOR_VERSION >= 3:
        _retry_windows_file_lock(lambda: group.update_attributes(merged))
    else:
        _retry_windows_file_lock(lambda: group.attrs.put(merged))


def write_zarr_region(array: zarr.Array, region, values) -> None:
    """Write a Zarr selection, retrying transient Windows file locks."""
    _retry_windows_file_lock(lambda: array.__setitem__(region, values))


def _reflected_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * size - 2
    folded = np.mod(indices, period)
    return np.where(folded < size, folded, period - folded).astype(np.int64)


def read_ndarray_box(array, start, end, *, pad_mode="reflect", constant_values=0):
    """Read possibly out-of-bounds coordinates from a NumPy array."""
    array = np.asarray(array)
    start = tuple(int(value) for value in start)
    end = tuple(int(value) for value in end)
    if pad_mode == "reflect":
        mapped = [
            _reflected_indices(np.arange(start[axis], end[axis], dtype=np.int64), array.shape[axis])
            for axis in range(3)
        ]
        result = array
        for axis, indices in enumerate(mapped):
            result = np.take(result, indices, axis=axis)
        return np.ascontiguousarray(result)

    clipped_start = tuple(max(0, value) for value in start)
    clipped_end = tuple(min(array.shape[axis], end[axis]) for axis in range(3))
    region = tuple(slice(clipped_start[axis], clipped_end[axis]) for axis in range(3))
    result = array[region]
    padding = tuple((clipped_start[a] - start[a], end[a] - clipped_end[a]) for a in range(3))
    return np.ascontiguousarray(np.pad(result, padding, mode="constant", constant_values=constant_values))


def iter_chunk_slices(shape: Sequence[int], chunks: Sequence[int]) -> Iterator[tuple[slice, ...]]:
    """Yield a deterministic grid of slices covering *shape*."""
    grid = [range(0, int(size), int(chunk)) for size, chunk in zip(shape, chunks)]
    for start in np.ndindex(*(len(axis) for axis in grid)):
        offsets = [grid[axis][index] for axis, index in enumerate(start)]
        yield tuple(slice(offset, min(offset + chunks[axis], shape[axis])) for axis, offset in enumerate(offsets))


def _storage_chunks(shape: Sequence[int], dtype: np.dtype, target_bytes: int = DEFAULT_STORAGE_CHUNK_BYTES) -> tuple[int, int, int]:
    """Choose spatial chunks that are small enough for random patch access."""
    if len(shape) != 3:
        raise ValueError(f"Penumbria only supports 3-D volumes, got shape {tuple(shape)}")
    # Start with microscopy-friendly Z/Y/X chunks, then shrink until they fit.
    itemsize = np.dtype(dtype).itemsize
    # TIFF conversion reads each Z plane once into an aligned slab.  Bound the
    # complete-XY slab to 512 MiB even for images around 3000 x 3000 pixels.
    slab_depth = max(1, (512 * 1024**2) // max(1, int(shape[1]) * int(shape[2]) * itemsize))
    chunks = [min(int(shape[0]), 64, slab_depth), min(int(shape[1]), 256), min(int(shape[2]), 256)]
    while math.prod(chunks) * itemsize > target_bytes:
        axis = int(np.argmax(chunks))
        chunks[axis] = max(1, chunks[axis] // 2)
    return tuple(chunks)


def _source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_name(path: Path, fingerprint: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf8")).hexdigest()[:12]
    return f"{path.stem}-{digest}.ome.zarr"


def _set_ngff_metadata(group: zarr.Group, name: str, is_label: bool = False) -> None:
    updates = {}
    if "multiscales" not in group.attrs:
        updates["multiscales"] = [{
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }]
    if is_label and "image-label" not in group.attrs:
        updates["image-label"] = {"version": "0.4"}
    if updates:
        update_group_attributes(group, updates)


def create_ome_zarr(
    path: os.PathLike[str] | str,
    shape: Sequence[int],
    dtype: np.dtype | str,
    *,
    chunks: Sequence[int] | None = None,
    name: str | None = None,
    is_label: bool = False,
    overwrite: bool = False,
) -> tuple[zarr.Group, zarr.Array]:
    """Create a level-0, compressed OME-Zarr array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    group = _open_group(path, mode)
    dtype = np.dtype(dtype)
    chunks = tuple(chunks or _storage_chunks(shape, dtype))
    if "0" in group:
        array = group["0"]
        if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != dtype:
            raise ValueError(f"Existing Zarr array at {path} has incompatible shape or dtype")
    else:
        array = create_zarr_array(
            group,
            "0",
            shape=tuple(int(v) for v in shape),
            chunks=chunks,
            dtype=dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            overwrite=False,
        )
    _set_ngff_metadata(group, name or path.stem, is_label=is_label)
    return group, array


def cache_tiff_as_ome_zarr(
    source_path: os.PathLike[str] | str,
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    output_dtype: np.dtype | str | None = None,
    is_label: bool = False,
) -> Path:
    """Convert a TIFF to a resumable local OME-Zarr cache and return its path."""
    source_path = Path(source_path)
    if source_path.suffix.lower() not in SUPPORTED_TIFF_SUFFIXES:
        raise ValueError(f"Unsupported TIFF extension: {source_path}")
    fingerprint = _source_fingerprint(source_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else source_path.parent / ".penumbria_zarr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / _cache_name(source_path, fingerprint)

    with tifffile.TiffFile(str(source_path)) as tiff:
        source = tiff.series[0]
        source_shape = tuple(int(value) for value in source.shape)
        if len(source_shape) != 3:
            raise ValueError(f"Only 3-D TIFF volumes are supported, got {source_shape} for {source_path}")
        dtype = np.dtype(output_dtype or source.dtype)
        chunks = _storage_chunks(source_shape, dtype)
        group, target = create_ome_zarr(
            destination, source_shape, dtype, chunks=chunks, name=source_path.stem, is_label=is_label
        )
        if group.attrs.get("penumbria_complete") and group.attrs.get("penumbria_source") == fingerprint:
            return destination

        grid_shape = tuple(math.ceil(size / chunk) for size, chunk in zip(source_shape, chunks))
        # Do not keep progress in a Zarr array.  Every scalar update rewrites
        # the same small chunk, and Zarr 3's atomic replacement can fail on
        # Windows when that destination file is briefly held open.  An append-
        # only sidecar retains resumability without replacing existing files.
        progress_path = destination.with_name(f".{destination.name}.progress")
        completion_grid = np.zeros(grid_shape, dtype=bool)
        if "_chunk_complete" in group:
            # Import progress from caches started by older Penumbria versions.
            try:
                completion_grid |= np.asarray(group["_chunk_complete"][:], dtype=bool)
            except (OSError, ValueError):
                pass
        if progress_path.exists():
            with progress_path.open("r", encoding="ascii") as progress_file:
                for line in progress_file:
                    try:
                        chunk_index = tuple(int(value) for value in line.strip().split(","))
                        if len(chunk_index) == 3 and all(
                            0 <= chunk_index[axis] < grid_shape[axis] for axis in range(3)
                        ):
                            completion_grid[chunk_index] = True
                    except ValueError:
                        # A final partial line after an interrupted write is safe
                        # to ignore; that target chunk will simply be rewritten.
                        continue
        total = math.prod(grid_shape)
        written = int(completion_grid.sum())
        print(f"Caching {source_path.name} as OME-Zarr ({written}/{total} chunks already present)...")

        pages_are_z_planes = (
            len(source.pages) == source_shape[0]
            and tuple(source.pages[0].shape) == tuple(source_shape[1:])
        )
        fallback_array = None
        if not pages_are_z_planes:
            try:
                fallback_array = tifffile.memmap(str(source_path), series=0, mode="r")
            except (ValueError, TypeError):
                # Handles unusual compressed single-page 3-D TIFF layouts using
                # a disk-backed temporary array rather than a full RAM copy.
                fallback_array = source.asarray(out="memmap")

        with progress_path.open("a", encoding="ascii") as progress_file:
            for z_index, z_start in enumerate(range(0, source_shape[0], chunks[0])):
                if completion_grid[z_index].all():
                    continue
                z_stop = min(z_start + chunks[0], source_shape[0])
                if pages_are_z_planes:
                    slab = np.empty((z_stop - z_start, source_shape[1], source_shape[2]), dtype=source.dtype)
                    for local_z, global_z in enumerate(range(z_start, z_stop)):
                        slab[local_z] = source.pages[global_z].asarray()
                else:
                    slab = np.asarray(fallback_array[z_start:z_stop])

                for y_index, y_start in enumerate(range(0, source_shape[1], chunks[1])):
                    y_stop = min(y_start + chunks[1], source_shape[1])
                    for x_index, x_start in enumerate(range(0, source_shape[2], chunks[2])):
                        chunk_index = (z_index, y_index, x_index)
                        if completion_grid[chunk_index]:
                            continue
                        x_stop = min(x_start + chunks[2], source_shape[2])
                        region = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
                        write_zarr_region(
                            target, region,
                            np.asarray(slab[:, y_start:y_stop, x_start:x_stop], dtype=dtype),
                        )
                        progress_file.write(",".join(str(value) for value in chunk_index) + "\n")
                        progress_file.flush()
                        completion_grid[chunk_index] = True

        update_group_attributes(group, {
            "penumbria_source": fingerprint,
            "penumbria_complete": True,
            "penumbria_working_memory_bytes": DEFAULT_WORKING_MEMORY_BYTES,
        })
        try:
            progress_path.unlink()
        except FileNotFoundError:
            pass
        return destination


def _float32_sort_keys(values: np.ndarray) -> np.ndarray:
    """Map IEEE float32 values to uint32 keys with the same total order."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(negative, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32, copy=False)


def _key_to_float32(key: int) -> float:
    key_u = np.uint32(key)
    if key_u & np.uint32(0x80000000):
        bits = key_u ^ np.uint32(0x80000000)
    else:
        bits = ~key_u
    return float(np.asarray(bits, dtype=np.uint32).view(np.float32))


def exact_float32_percentiles(array: zarr.Array, percentiles: Sequence[float]) -> list[float]:
    """Compute NumPy-compatible linear percentiles in two bounded-memory passes.

    A radix histogram avoids the full-sized temporary copy made by
    ``np.percentile``.  It is exact for the float32 values stored by the data
    preparation pipeline.
    """
    if array.ndim != 3:
        raise ValueError("Percentile calculation expects a 3-D array")
    count = int(math.prod(array.shape))
    if count == 0:
        raise ValueError("Cannot normalise an empty image")
    requested_ranks: set[int] = set()
    rank_info: list[tuple[int, int, float]] = []
    for percentile in percentiles:
        position = (count - 1) * (float(percentile) / 100.0)
        lower, upper = int(math.floor(position)), int(math.ceil(position))
        requested_ranks.update((lower, upper))
        rank_info.append((lower, upper, position - lower))

    upper_hist = np.zeros(1 << 16, dtype=np.uint64)
    for region in iter_chunk_slices(array.shape, array.chunks):
        values = np.asarray(array[region], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Images containing NaN or infinity cannot be globally normalised")
        upper = (_float32_sort_keys(values.ravel()) >> np.uint32(16)).astype(np.int32, copy=False)
        upper_hist += np.bincount(upper, minlength=1 << 16).astype(np.uint64)

    cumulative = np.cumsum(upper_hist, dtype=np.uint64)
    rank_buckets: dict[int, tuple[int, int]] = {}
    for rank in requested_ranks:
        bucket = int(np.searchsorted(cumulative, rank + 1, side="left"))
        before = int(cumulative[bucket - 1]) if bucket else 0
        rank_buckets[rank] = (bucket, rank - before)

    needed_buckets = sorted({bucket for bucket, _ in rank_buckets.values()})
    lower_hists = {bucket: np.zeros(1 << 16, dtype=np.uint64) for bucket in needed_buckets}
    for region in iter_chunk_slices(array.shape, array.chunks):
        keys = _float32_sort_keys(np.asarray(array[region], dtype=np.float32).ravel())
        upper = keys >> np.uint32(16)
        for bucket in needed_buckets:
            lower = (keys[upper == bucket] & np.uint32(0xFFFF)).astype(np.int32, copy=False)
            if lower.size:
                lower_hists[bucket] += np.bincount(lower, minlength=1 << 16).astype(np.uint64)

    rank_values: dict[int, float] = {}
    for rank, (bucket, offset) in rank_buckets.items():
        lower = int(np.searchsorted(np.cumsum(lower_hists[bucket], dtype=np.uint64), offset + 1, side="left"))
        rank_values[rank] = _key_to_float32((bucket << 16) | lower)

    result = []
    for lower, upper, fraction in rank_info:
        low_value, high_value = rank_values[lower], rank_values[upper]
        result.append(low_value + (high_value - low_value) * fraction)
    return result


@dataclass(frozen=True)
class NormalizationStats:
    low_percentile: float
    high_percentile: float
    low_value: float
    high_value: float


class ChunkedVolume:
    """Small array-like facade over a level-0 Zarr image."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        normalization: NormalizationStats | None = None,
        read_dtype: np.dtype | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.group = _open_group(self.path, "r")
        self.array = self.group["0"]
        self.normalization = normalization
        self.read_dtype = np.dtype(read_dtype) if read_dtype is not None else None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        if self.read_dtype is not None:
            return self.read_dtype
        if self.normalization is not None:
            return np.dtype(np.float32)
        return np.dtype(self.array.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(self.array.chunks)

    def _transform(self, values: np.ndarray) -> np.ndarray:
        if self.normalization is not None:
            stats = self.normalization
            values = np.asarray(values, dtype=np.float32)
            if stats.high_value > stats.low_value:
                values = np.clip(values, stats.low_value, stats.high_value)
                values = (values - stats.low_value) / (stats.high_value - stats.low_value)
            else:
                values = np.ones_like(values, dtype=np.float32) * stats.low_value
        if self.read_dtype is not None:
            values = values.astype(self.read_dtype, copy=False)
        return np.asarray(values)

    def __getitem__(self, key):
        return self._transform(np.asarray(self.array[key]))

    def read_region(self, region: Sequence[slice]) -> np.ndarray:
        return self[tuple(region)]

    def read_box(
        self,
        start: Sequence[int],
        end: Sequence[int],
        *,
        pad_mode: str = "reflect",
        constant_values: float | int = 0,
    ) -> np.ndarray:
        """Read global half-open bounds and pad only the returned small block."""
        start = tuple(int(v) for v in start)
        end = tuple(int(v) for v in end)
        if pad_mode == "reflect":
            mapped = [
                _reflected_indices(np.arange(start[axis], end[axis], dtype=np.int64), self.shape[axis])
                for axis in range(3)
            ]
            source_start = tuple(int(indices.min()) for indices in mapped)
            source_end = tuple(int(indices.max()) + 1 for indices in mapped)
            region = tuple(slice(source_start[axis], source_end[axis]) for axis in range(3))
            values = self.read_region(region)
            for axis, indices in enumerate(mapped):
                values = np.take(values, indices - source_start[axis], axis=axis)
            return np.ascontiguousarray(values)

        clipped_start = tuple(max(0, value) for value in start)
        clipped_end = tuple(min(self.shape[axis], value) for axis, value in enumerate(end))
        region = tuple(slice(clipped_start[axis], clipped_end[axis]) for axis in range(3))
        values = self.read_region(region)
        pad_width = tuple(
            (clipped_start[axis] - start[axis], end[axis] - clipped_end[axis]) for axis in range(3)
        )
        if any(before or after for before, after in pad_width):
            values = np.pad(values, pad_width, mode="constant", constant_values=constant_values)
        return np.ascontiguousarray(values)

    def read_patch(
        self,
        center: Sequence[int],
        patch_shape: Sequence[int],
        *,
        pad_mode: str = "reflect",
        constant_values: float | int = 0,
    ) -> np.ndarray:
        patch_shape = tuple(int(v) for v in patch_shape)
        start = tuple(int(center[axis]) - patch_shape[axis] // 2 for axis in range(3))
        end = tuple(start[axis] + patch_shape[axis] for axis in range(3))
        return self.read_box(start, end, pad_mode=pad_mode, constant_values=constant_values)


def open_cached_volume(
    source_path: os.PathLike[str] | str,
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    storage_dtype: np.dtype | str | None = None,
    read_dtype: np.dtype | str | None = None,
    normalize: bool = False,
    low_percentile: float = 1.0,
    high_percentile: float = 99.9,
    is_label: bool = False,
) -> ChunkedVolume:
    """Open a TIFF through its OME-Zarr cache, optionally globally normalised."""
    cache_path = cache_tiff_as_ome_zarr(
        source_path, cache_dir=cache_dir, output_dtype=storage_dtype, is_label=is_label
    )
    group = _open_group(cache_path, "a")
    stats = None
    if normalize:
        key = f"{low_percentile:g}_{high_percentile:g}"
        cached_stats = dict(group.attrs.get("penumbria_normalization", {}))
        if key not in cached_stats:
            print(f"Computing exact whole-volume percentiles for {Path(source_path).name}...")
            low_value, high_value = exact_float32_percentiles(group["0"], [low_percentile, high_percentile])
            cached_stats[key] = {
                "low_percentile": float(low_percentile),
                "high_percentile": float(high_percentile),
                "low_value": float(low_value),
                "high_value": float(high_value),
            }
            update_group_attributes(group, {"penumbria_normalization": cached_stats})
        stats = NormalizationStats(**cached_stats[key])
    return ChunkedVolume(cache_path, normalization=stats, read_dtype=read_dtype)


def open_ome_zarr(path: os.PathLike[str] | str, *, read_dtype=None) -> ChunkedVolume:
    return ChunkedVolume(path, read_dtype=read_dtype)


def imagej_label_dtype(max_label: int) -> np.dtype:
    """Return an ImageJ-native dtype that represents all label IDs exactly."""
    max_label = int(max_label)
    if max_label < 0:
        raise ValueError("Label IDs must be non-negative")
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    # ImageJ has no native uint32 image type. Float32 exactly represents every
    # integer through 2**24 and is therefore the lossless fallback.
    if max_label <= 2**24:
        return np.dtype(np.float32)
    raise ValueError(
        f"ImageJ TIFF cannot exactly represent label ID {max_label}; "
        "keep the uint32 OME-Zarr output instead"
    )


def export_zarr_to_tiff(
    volume: ChunkedVolume,
    destination: os.PathLike[str] | str,
    *,
    output_dtype: np.dtype | str | None = None,
) -> Path:
    """Stream a Zarr volume to a native ImageJ hyperstack TIFF."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(output_dtype or volume.dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        raise ValueError(
            f"{dtype} is not a native ImageJ image type; pass uint8, uint16, or float32 as output_dtype"
        )

    def planes() -> Iterable[np.ndarray]:
        for z_index in range(volume.shape[0]):
            yield np.asarray(volume[z_index], dtype=dtype)

    tifffile.imwrite(
        str(destination),
        data=planes(),
        shape=volume.shape,
        dtype=dtype,
        # ImageJ hyperstacks intentionally support data beyond classic TIFF's
        # 4 GiB offset limit by storing contiguous data after the first IFD.
        bigtiff=False,
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    return destination


@dataclass(frozen=True)
class SamplingIndex:
    object_centres: np.ndarray
    difficult_background: np.ndarray
    difficult_foreground: np.ndarray


def _empty_locations() -> np.ndarray:
    return np.empty((0, 3), dtype=np.int64)


def build_sampling_index(
    integer_labels: ChunkedVolume,
    image: ChunkedVolume,
    patch_shape: Sequence[int],
) -> SamplingIndex:
    """Build and cache bounded-memory sampling metadata for a training pair.

    Object locations use the centre of each instance bounding box.  Difficult
    examples use one smoothed intensity extremum per patch-sized spatial bin,
    which closely follows the old global maximum/minimum-filter sampler without
    allocating several complete volumes.
    """
    patch_shape = tuple(int(v) for v in patch_shape)
    index_path = integer_labels.path.parent / (
        f"{integer_labels.path.stem}.sampling-{'x'.join(map(str, patch_shape))}.npz"
    )
    if index_path.exists():
        cached = np.load(index_path)
        return SamplingIndex(
            np.asarray(cached["object_centres"], dtype=np.int64),
            np.asarray(cached["difficult_background"], dtype=np.int64),
            np.asarray(cached["difficult_foreground"], dtype=np.int64),
        )

    print(f"Building out-of-core sampling index for {integer_labels.path.name}...")
    scan_chunks = tuple(min(size, chunk) for size, chunk in zip(integer_labels.shape, (96, 256, 256)))
    boxes: dict[int, list[np.ndarray]] = {}
    difficult_background: list[np.ndarray] = []
    difficult_foreground: list[np.ndarray] = []
    bin_shape = tuple(max(16, value) for value in patch_shape)
    gaussian_halo = 8  # scipy's sigma=2 default support is four sigma

    for region in iter_chunk_slices(integer_labels.shape, scan_chunks):
        starts = np.asarray([sl.start for sl in region], dtype=np.int64)
        ends = np.asarray([sl.stop for sl in region], dtype=np.int64)
        labels = np.asarray(integer_labels.read_region(region), dtype=np.int64)
        labels[labels < 0] = 0

        for label_id in np.unique(labels):
            if label_id <= 0:
                continue
            locations = np.argwhere(labels == label_id)
            local_min = locations.min(axis=0) + starts
            local_max = locations.max(axis=0) + starts
            if int(label_id) not in boxes:
                boxes[int(label_id)] = [local_min, local_max]
            else:
                boxes[int(label_id)][0] = np.minimum(boxes[int(label_id)][0], local_min)
                boxes[int(label_id)][1] = np.maximum(boxes[int(label_id)][1], local_max)

        expanded = image.read_box(starts - gaussian_halo, ends + gaussian_halo, pad_mode="reflect")
        smoothed = gaussian_filter(np.asarray(expanded, dtype=np.float32), sigma=2)
        core_shape = tuple(int(ends[axis] - starts[axis]) for axis in range(3))
        core = smoothed[
            gaussian_halo:gaussian_halo + core_shape[0],
            gaussian_halo:gaussian_halo + core_shape[1],
            gaussian_halo:gaussian_halo + core_shape[2],
        ]

        for local_start in np.ndindex(*(math.ceil(core_shape[a] / bin_shape[a]) for a in range(3))):
            block_start = np.asarray([local_start[a] * bin_shape[a] for a in range(3)], dtype=np.int64)
            block_end = np.minimum(block_start + np.asarray(bin_shape), np.asarray(core_shape))
            block_slice = tuple(slice(int(block_start[a]), int(block_end[a])) for a in range(3))
            label_block = labels[block_slice]
            image_block = core[block_slice]

            background = label_block == 0
            if np.any(background):
                scores = np.where(background, image_block, -np.inf)
                flat = int(np.argmax(scores))
                if np.isfinite(scores.ravel()[flat]) and scores.ravel()[flat] > 0:
                    difficult_background.append(starts + block_start + np.asarray(np.unravel_index(flat, scores.shape)))

            foreground = label_block > 0
            if np.any(foreground):
                scores = np.where(foreground, image_block, np.inf)
                flat = int(np.argmin(scores))
                if np.isfinite(scores.ravel()[flat]):
                    difficult_foreground.append(starts + block_start + np.asarray(np.unravel_index(flat, scores.shape)))

    object_centres = np.asarray(
        [(bounds[0] + bounds[1]) // 2 for _, bounds in sorted(boxes.items())], dtype=np.int64
    ) if boxes else _empty_locations()
    background_array = np.asarray(difficult_background, dtype=np.int64) if difficult_background else _empty_locations()
    foreground_array = np.asarray(difficult_foreground, dtype=np.int64) if difficult_foreground else _empty_locations()
    np.savez_compressed(
        index_path,
        object_centres=object_centres,
        difficult_background=background_array,
        difficult_foreground=foreground_array,
    )
    return SamplingIndex(object_centres, background_array, foreground_array)


def resize_volume_to_ome_zarr(
    volume: ChunkedVolume,
    output_path: os.PathLike[str] | str,
    output_shape: Sequence[int],
    *,
    order: int = 3,
    is_label: bool = False,
) -> ChunkedVolume:
    """Resize a 3-D volume chunk by chunk using globally aligned coordinates."""
    if any(int(value) != value for value in output_shape):
        raise ValueError("output_shape must contain integer voxel counts; use scaled_shape() first")
    output_shape = tuple(max(1, int(value)) for value in output_shape)
    output_dtype = np.int32 if is_label else np.float32
    _, output = create_ome_zarr(
        output_path, output_shape, output_dtype,
        name=Path(output_path).stem, is_label=is_label, overwrite=True,
    )
    scale = np.asarray(volume.shape, dtype=np.float64) / np.asarray(output_shape, dtype=np.float64)
    halo = max(2, int(order) + 1)
    for region in iter_chunk_slices(output_shape, output.chunks):
        output_axes = [np.arange(sl.start, sl.stop, dtype=np.float64) for sl in region]
        source_axes = [(axis + 0.5) * scale[index] - 0.5 for index, axis in enumerate(output_axes)]
        source_start = np.asarray([math.floor(axis.min()) - halo for axis in source_axes], dtype=np.int64)
        source_end = np.asarray([math.ceil(axis.max()) + halo + 1 for axis in source_axes], dtype=np.int64)
        source_block = volume.read_box(source_start, source_end, pad_mode="reflect")
        coordinates = np.meshgrid(*[
            axis - source_start[index] for index, axis in enumerate(source_axes)
        ], indexing="ij")
        resized = map_coordinates(
            np.asarray(source_block), coordinates, order=int(order), mode="reflect", prefilter=order > 1
        )
        if is_label:
            resized = np.rint(resized).astype(np.int32)
        write_zarr_region(output, region, np.asarray(resized, dtype=output_dtype))
    return ChunkedVolume(output_path)
