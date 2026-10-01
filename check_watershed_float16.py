"""Does float16 heat change the watershed on your heatmaps?

The watershed only compares and moves heat values, it never computes with them, so storing the heat
in float16 gives exactly the labels of the float32 watershed run on the heat rounded to float16.
This script runs both on a predicted heatmap and reports what changes.

    python check_watershed_float16.py path/to/heatmap.tif --prominence 0.23 --background 0.06
"""
import argparse

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, label

from gpu_patched import h_dome, seeded_watershed


def seeds_of(prediction, prominence):
    """The seeds of postprocess.watershed_inference with gaussian smoothing on."""
    smoothed = gaussian_filter(prediction, 1)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    return label((h_dome(smoothed, prominence) > 0) & (prediction > prominence))


def equal_neighbour_share(values, foreground):
    """Share of foreground voxels that have a face neighbour with exactly the same value."""
    tied = np.zeros(values.shape, bool)
    for axis in range(3):
        low = tuple(slice(0, -1) if a == axis else slice(None) for a in range(3))
        high = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        equal = (values[low] == values[high]) & foreground[low] & foreground[high]
        tied[low] |= equal
        tied[high] |= equal
    return tied[foreground].mean()


def report(prediction, prominence, background):
    seeds, count = seeds_of(prediction, prominence)
    rounded = prediction.astype(np.float16).astype(np.float32)
    foreground = prediction > background
    flips = int(((rounded > background) != foreground).sum())
    print(f"{prediction.size / 1e6:.0f}M voxels, {count:,} seeds, {foreground.mean():.1%} foreground")
    print(f"voxels crossing the background threshold: {flips:,} ({flips / foreground.sum():.4%} of the foreground)")
    print(f"foreground voxels with an equal neighbour: float32 {equal_neighbour_share(prediction, foreground):.4%}"
          f" -> float16 {equal_neighbour_share(rounded, foreground):.4%}")

    exact = seeded_watershed(prediction, seeds, background)
    half = seeded_watershed(rounded, seeds, background)
    differing = int((exact != half).sum())
    print(f"voxels labelled differently: {differing:,} ({differing / (exact > 0).sum():.4%} of the labelled voxels)")
    sizes = np.bincount(exact.ravel(), minlength=count + 1)[1:]
    half_sizes = np.bincount(half.ravel(), minlength=count + 1)[1:]
    change = np.abs(half_sizes - sizes)
    cells = sizes > 1000
    relative = change[cells] / sizes[cells]
    print(f"cells over 1000 voxels: {cells.sum():,}; volume change per cell: median {np.median(relative):.3%}, "
          f"95th percentile {np.percentile(relative, 95):.2%}, largest {relative.max():.1%}")
    for index in np.argsort(-change)[:3]:
        print(f"  label {index + 1}: {sizes[index]:,} -> {half_sizes[index]:,} voxels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("heatmap", help="predicted heatmap as a TIFF")
    parser.add_argument("--prominence", type=float, default=0.23, help="cell_prominence (h of the h-dome)")
    parser.add_argument("--background", type=float, default=0.06, help="background_threshold")
    args = parser.parse_args()
    report(tifffile.imread(args.heatmap).astype(np.float32), args.prominence, args.background)
