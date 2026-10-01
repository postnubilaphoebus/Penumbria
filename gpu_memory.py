"""How much GPU memory the watershed may use, measured at run time.

Free memory is read when a watershed starts, and at most 70% of it is used. How many
voxels that buys is measured, not assumed: `calibrate` finds the volume size at which
this GPU actually runs out of memory in `seeded_watershed_gpu`, once per GPU, and caches
the result. Run `python gpu_memory.py` to see the numbers or `--recalibrate` to redo them.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pynvml
import taichi as ti
import torch

from gpu_morphology import F32, I32, frees_gpu_memory_here, init_cuda
from gpu_watershed import seeded_watershed_gpu
from patch_plan import MAX_VOXELS, PatchBudget, is_gpu_oom

log = logging.getLogger("penumbria.gpu")

BUDGET_FRACTION = 0.7
# Six 4-byte arrays are alive at the peak of the watershed (heat, seeds, levels, parent,
# distance, flat list), so this is the raw size of a voxel; the measured overhead is on top.
WATERSHED_BYTES_PER_VOXEL = 24
CALIBRATION_FILE = Path.home() / ".penumbria" / "gpu_calibration.json"
TRIAL_TIMEOUT_SECONDS = 180  # a trial that spills into system memory instead of failing is far slower than this
OOM_EXIT_CODE = 3


def free_gpu_bytes():
    """GPU memory this process can use right now, counting what torch has cached and Taichi has freed.

    Two readings, and the smaller one counts. The driver's (torch.cuda.mem_get_info) is what this
    process can allocate; on Windows it is capped by the process's share of the memory but cannot
    see how much other programs use. NVML sees the whole device, other programs included, but
    does not know about that cap.
    """
    init_cuda()
    ti.sync()  # Taichi returns the memory of freed arrays to the driver at a sync
    torch.cuda.empty_cache()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    return min(torch.cuda.mem_get_info()[0], pynvml.nvmlDeviceGetMemoryInfo(handle).free)


def device_key():
    properties = torch.cuda.get_device_properties(0)
    return f"{properties.name} / {properties.total_memory}"


@ti.kernel
def _fill_synthetic(heat: F32, seeds: I32):
    for z, y, x in heat:
        value = 0.5 * ti.sin(z / 9.0) * ti.sin(y / 7.0) * ti.sin(x / 11.0) + 0.5
        heat[z, y, x] = value
        seeds[z, y, x] = 0
        if z % 14 == 4 and y % 14 == 4 and x % 14 == 4 and value > 0.8:
            seeds[z, y, x] = (z * heat.shape[1] + y) * heat.shape[2] + x + 1


@frees_gpu_memory_here
def _watershed_runs(voxels):
    """Whether the full watershed fits on the GPU for a cube of about `voxels` voxels."""
    side = max(2, round(voxels ** (1 / 3)))
    heat, seeds = ti.ndarray(ti.f32, (side,) * 3), ti.ndarray(ti.i32, (side,) * 3)
    _fill_synthetic(heat, seeds)
    seeded_watershed_gpu(heat, seeds, 0.1)
    ti.sync()
    return True


def find_oom_voxels(fits, start, tolerance=0.02):
    """The largest voxel count for which `fits(n)` is true, to within `tolerance` (bracket, then bisect)."""
    low = high = start
    if fits(start):
        while True:  # grow until it no longer fits
            low = high
            if high >= MAX_VOXELS:
                return MAX_VOXELS
            high = min(MAX_VOXELS, int(high * 1.5))
            if not fits(high):
                break
    else:
        while True:  # shrink until it fits
            high = low
            low //= 2
            if low < 1:
                raise RuntimeError("The watershed does not even fit a single voxel on this GPU")
            if fits(low):
                break
    while high - low > 1 and high > low * (1 + tolerance):  # fits(low) and not fits(high)
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def _run_trial(voxels):
    """(fits, free GPU bytes before the run) for one watershed of `voxels` voxels in a fresh process.

    A fresh process is needed because after one failed allocation, CUDA/Taichi on some systems
    keeps allocating beyond the GPU's memory (spilling into system RAM, ten times slower or worse),
    which hides the real limit. A trial that times out is counted as not fitting.
    """
    try:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trial", str(voxels)],
                              capture_output=True, text=True, timeout=TRIAL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False, None
    if done.returncode not in (0, OOM_EXIT_CODE):
        raise RuntimeError(f"GPU memory trial of {voxels:,} voxels failed: {done.stderr[-2000:]}")
    free = next((int(line.split()[1]) for line in done.stdout.splitlines() if line.startswith("FREE ")), None)
    return done.returncode == 0, free


def calibrate():
    """Find this GPU's out-of-memory point for the watershed and cache the overhead it implies."""
    init_cuda()
    log.info("Calibrating GPU memory use of the watershed (one time, cached in %s)...", CALIBRATION_FILE)
    started, frees = time.perf_counter(), []

    def fits(voxels):
        ok, free = _run_trial(voxels)
        frees.append(free)
        log.info("  %s voxels: %s", f"{voxels:,}", "fits" if ok else "out of memory")
        return ok

    oom_voxels = find_oom_voxels(fits, start=int(free_gpu_bytes() / WATERSHED_BYTES_PER_VOXEL * 0.8))
    free = max(f for f in frees if f)  # what a fresh process sees, before its own allocations
    record = {"free_bytes": int(free), "oom_voxels": int(oom_voxels),
              "bytes_per_voxel": free / oom_voxels,
              "overhead": max(1.0, free / oom_voxels / WATERSHED_BYTES_PER_VOXEL)}
    cache = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
    cache[device_key()] = record
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(cache, indent=2))
    log.info("Out of memory above %s voxels with %.2f GiB free (%.1f bytes/voxel, found in %.0f s)",
             f"{oom_voxels:,}", free / 2**30, record["bytes_per_voxel"], time.perf_counter() - started)
    return record


def cached_calibration():
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text()).get(device_key())
    return None


def watershed_budget(voxels):
    """Budget of 70% of the GPU memory free now, for a watershed of `voxels` voxels.

    The measured overhead is applied when this GPU has been calibrated. A volume that fits
    with the plain array sizes does not trigger a calibration; a larger one does, once.
    """
    init_cuda()
    budget = int(BUDGET_FRACTION * free_gpu_bytes())
    record = cached_calibration()
    if record is None and voxels * WATERSHED_BYTES_PER_VOXEL > budget:
        record = calibrate()
    return PatchBudget(budget, record["overhead"] if record else 1.0, free_bytes=free_gpu_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Show or redo the GPU memory calibration of the watershed")
    parser.add_argument("--recalibrate", action="store_true", help="measure again even if a result is cached")
    parser.add_argument("--trial", type=int, help=argparse.SUPPRESS)  # one calibration trial, run by calibrate()
    args = parser.parse_args()
    init_cuda()
    if args.trial:
        print(f"FREE {free_gpu_bytes()}", flush=True)
        try:
            _watershed_runs(args.trial)
        except RuntimeError as error:
            if not is_gpu_oom(error):
                raise
            sys.exit(OOM_EXIT_CODE)
        sys.exit(0)
    record = calibrate() if args.recalibrate or cached_calibration() is None else cached_calibration()
    free = free_gpu_bytes()
    voxels = int(BUDGET_FRACTION * free / (WATERSHED_BYTES_PER_VOXEL * record["overhead"]))
    print(f"GPU: {device_key()}")
    print(f"free now: {free / 2**30:.2f} GiB -> budget {BUDGET_FRACTION:.0%} = {BUDGET_FRACTION * free / 2**30:.2f} GiB")
    print(f"measured out-of-memory point: {record['oom_voxels']:,} voxels "
          f"({record['bytes_per_voxel']:.1f} bytes/voxel, {record['free_bytes'] / 2**30:.2f} GiB free then)")
    print(f"working patch size now: {voxels:,} voxels")
