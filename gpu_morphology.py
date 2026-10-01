"""GPU (Taichi/CUDA) grayscale reconstruction by dilation, 6-connectivity.

The result is the unique fixed point of m <- min(dilate(m), mask). Only
min/max are involved, so update order does not matter and passes run in place
without atomics. Iteration stops after a batch of passes with no change, so
the result is an exact fixed point. (SimpleITK's is not: it leaves some voxels
one ulp low, which become spurious single-voxel h-maxima.) Voxels with
marker >= mask are already final and are skipped.

The numpy wrappers (h_dome, reconstruct_by_dilation) start CUDA on first use
and accept 2-D images as single-slice volumes, where 6-connectivity is the
usual 4-connectivity.
"""
import functools
import gc

import numpy as np
import taichi as ti

F32 = ti.types.ndarray(dtype=ti.f32, ndim=3)
I32 = ti.types.ndarray(dtype=ti.i32, ndim=3)
LIST = ti.types.ndarray(dtype=ti.i32, ndim=1)
NEIGHBOURS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
_cuda_ready = False


def init_cuda(**kwargs):
    """ti.init on CUDA once per process; raises instead of Taichi's silent
    fallback to the CPU. Another ti.init call elsewhere resets Taichi."""
    global _cuda_ready
    if _cuda_ready:
        return
    ti.init(arch=ti.cuda, **kwargs)
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("CUDA is unavailable; refusing to run on the CPU")
    _cuda_ready = True


def frees_gpu_memory_here(function):
    """Taichi arrays passed to kernels end up in reference cycles, and Taichi
    aborts if the garbage collector frees them on another thread (e.g. zarr's
    decoder threads). Pause automatic collection during the call and collect
    on the calling thread before returning."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        enabled = gc.isenabled()
        gc.disable()
        try:
            return function(*args, **kwargs)
        finally:
            gc.collect()
            if enabled:
                gc.enable()
    return wrapper


def to_device(array, dtype):
    """numpy (2-D or 3-D) -> 3-D ti.ndarray; 2-D gets a leading axis of 1."""
    init_cuda()
    array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim == 2:
        array = array[None]
    device = ti.ndarray(ti.f32 if dtype == np.float32 else ti.i32, array.shape)
    device.from_numpy(array)
    return device


@ti.func
def index_of(a: ti.template(), z, y, x):
    return (z * a.shape[1] + y) * a.shape[2] + x


@ti.func
def voxel_of(a: ti.template(), i):
    return i // (a.shape[1] * a.shape[2]), (i // a.shape[2]) % a.shape[1], i % a.shape[2]


@ti.func
def inside(a: ti.template(), z, y, x):
    return 0 <= z < a.shape[0] and 0 <= y < a.shape[1] and 0 <= x < a.shape[2]


def check_volume(*arrays):
    shape = arrays[0].shape
    if len(shape) != 3 or any(a.shape != shape for a in arrays):
        raise ValueError("inputs must be 3-D with identical shapes")
    if np.prod(shape, dtype=np.int64) >= 2**31:
        raise ValueError("volumes of 2**31 voxels or more need 64-bit indices")


def until_stable(step, check_every=8):
    """Call step(flag, i) for i = 1, 2, ... in batches of check_every until a
    batch leaves flag at 0. Returns the number of calls."""
    flag = ti.ndarray(ti.i32, 1)
    calls = 0
    while True:
        flag.fill(0)
        for _ in range(check_every):
            calls += 1
            step(flag, calls)
        if flag.to_numpy()[0] == 0:
            return calls


@ti.func
def _raise_to_neighbours(m: ti.template(), g: ti.template(), z, y, x, changed: ti.template()):
    value, ceiling = m[z, y, x], g[z, y, x]
    if value < ceiling:
        best = value
        for dz, dy, dx in ti.static(NEIGHBOURS):
            if inside(m, z + dz, y + dy, x + dx):
                best = ti.max(best, m[z + dz, y + dy, x + dx])
        best = ti.min(best, ceiling)
        if best > value:
            m[z, y, x] = best
            changed[0] = 1


@ti.kernel
def _clamp_below(m: F32, g: F32):
    for z, y, x in m:
        m[z, y, x] = ti.min(m[z, y, x], g[z, y, x])


@ti.kernel
def _list_active(m: F32, g: F32, active: LIST, count: LIST):
    for z, y, x in m:
        if m[z, y, x] < g[z, y, x]:
            active[ti.atomic_add(count[0], 1)] = index_of(m, z, y, x)


@ti.kernel
def _grid_pass(m: F32, g: F32, changed: LIST):
    for z, y, x in m:
        _raise_to_neighbours(m, g, z, y, x, changed)


@ti.kernel
def _list_pass(m: F32, g: F32, active: LIST, n: ti.i32, changed: LIST):
    for i in range(n):
        z, y, x = voxel_of(m, active[i])
        _raise_to_neighbours(m, g, z, y, x, changed)


def reconstruct_by_dilation_(m, g, *, compact=None, check_every=8):
    """In place on device arrays: m (marker, becomes the result) under g (mask).

    compact=True iterates only voxels with marker < mask, False the full grid,
    None the list when under half the voxels are active. Returns passes run.
    """
    check_volume(m, g)
    _clamp_below(m, g)
    if compact is not False:
        active, count = ti.ndarray(ti.i32, m.shape[0] * m.shape[1] * m.shape[2]), ti.ndarray(ti.i32, 1)
        _list_active(m, g, active, count)
        n = int(count.to_numpy()[0])
        if compact or 2 * n < active.shape[0]:
            return until_stable(lambda flag, _: _list_pass(m, g, active, n, flag), check_every)
        del active
    return until_stable(lambda flag, _: _grid_pass(m, g, flag), check_every)


@ti.kernel
def _subtract(a: F32, h: ti.f32, out: F32):
    for z, y, x in a:
        out[z, y, x] = a[z, y, x] - h


@ti.kernel
def _subtract_from(a: F32, out: F32):
    for z, y, x in a:
        out[z, y, x] = a[z, y, x] - out[z, y, x]


@frees_gpu_memory_here
def reconstruct_by_dilation(marker, mask, **kwargs):
    """numpy float32 in and out; see reconstruct_by_dilation_ for options."""
    m, g = to_device(marker, np.float32), to_device(mask, np.float32)
    reconstruct_by_dilation_(m, g, **kwargs)
    return m.to_numpy().reshape(np.shape(marker))


@frees_gpu_memory_here
def h_dome(image, h):
    """h-dome transform image - R(image - h), numpy float32 in and out.
    Voxels > 0 are the h-maxima (regional maxima of dynamic >= h)."""
    f = to_device(image, np.float32)
    m = ti.ndarray(ti.f32, f.shape)
    _subtract(f, h, m)
    reconstruct_by_dilation_(m, f)
    _subtract_from(f, m)
    return m.to_numpy().reshape(np.shape(image))
