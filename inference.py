import torch
import numpy as np
from tqdm import tqdm
import os
import warnings
import math
from contextlib import nullcontext
from scipy.ndimage import distance_transform_edt
from numcodecs import Blosc
from volume_io import (
    ChunkedVolume,
    create_ome_zarr,
    create_zarr_array,
    export_zarr_to_tiff,
    iter_chunk_slices,
    read_ndarray_box,
    update_group_attributes,
    write_zarr_region,
)


def get_autocast(mixed_precision=True):
    if not mixed_precision:
        return nullcontext()
    # New API (PyTorch 2.0+)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)

    # Old API (PyTorch ≤1.13)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def center_weighted_array(shape, feather_width = 32):
    """
    Returns a NumPy array of given shape with:
    - maximum value at the center
    - 0 at the outermost voxel layer
    - decaying with Euclidean distance from the outermost layer
    """
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = False  
    edt = distance_transform_edt(~mask)
    edt[mask] = 0
    max_val = edt.max()
    if max_val > 0:
        edt = edt / max_val
    else:
        edt = np.zeros_like(edt)
    return edt

def sliding_window_inference(model,
                             inference_images, 
                             masks_provided, 
                             mask_file_matrix,
                             mask_filename_matrix, 
                             device, 
                             low_value, 
                             high_value, 
                             predicted_label_path,
                             inference_filenames,
                             mixed_precision,
                             patch_based_norm = False,
                             tta = False,
                             image_dim = None,
                             keep_size = None,
                             step_size = None,
                             save_files = True):
    
    if not isinstance(inference_images, list):
        raise TypeError("inference_images must be a list, even when it contains one image")
    if len(image_dim) != 3 or len(keep_size) != 3 or len(step_size) != 3:
        raise ValueError("Out-of-core inference supports 3-D images only")
    if any(step_size[axis] > keep_size[axis] for axis in range(3)):
        raise ValueError("Each step size must be less than or equal to its keep size")
    if patch_based_norm:
        warnings.warn(
            "Patch-based normalisation is disabled: all patches use cached whole-image clipping and normalisation statistics."
        )

    output_directory = os.path.join(predicted_label_path, "preds")
    os.makedirs(output_directory, exist_ok=True)
    print(f"{'with' if tta else 'without'} test time augmentation (TTA)")
    print("with disk-backed OME-Zarr accumulation and Euclidean feathering")

    outputs = []
    output_names = inference_filenames or [f"volume_{index}" for index in range(len(inference_images))]
    weights = center_weighted_array(tuple(image_dim)).astype(np.float32)
    # Keep non-zero support at the model-patch boundary.  This only matters for
    # images smaller than keep_size in one dimension.
    weights = np.maximum(weights, np.float32(1e-6))

    for image_index, (image, output_name) in enumerate(zip(inference_images, output_names)):
        if len(image.shape) != 3:
            raise ValueError(f"Only 3-D inference images are supported, got {image.shape}")
        output_path = os.path.join(output_directory, f"{output_name}_inference_output.ome.zarr")
        group, output_array = create_ome_zarr(
            output_path,
            image.shape,
            np.float32,
            name=f"{output_name} Penumbria heatmap",
            overwrite=True,
        )
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
        sum_array = create_zarr_array(
            group, "_sum", shape=image.shape, chunks=output_array.chunks, dtype=np.float32,
            compressor=compressor, fill_value=0, overwrite=True,
        )
        weight_array = create_zarr_array(
            group, "_weights", shape=image.shape, chunks=output_array.chunks, dtype=np.float32,
            compressor=compressor, fill_value=0, overwrite=True,
        )

        pad = tuple((image_dim[axis] - keep_size[axis]) // 2 for axis in range(3))
        positions = [range(0, int(image.shape[axis]), int(step_size[axis])) for axis in range(3)]
        total_patches = math.prod(len(axis_positions) for axis_positions in positions)
        progress = tqdm(total=total_patches, desc=f"Inference {output_name}")
        for keep_start in np.ndindex(*(len(axis_positions) for axis_positions in positions)):
            global_keep_start = tuple(positions[axis][keep_start[axis]] for axis in range(3))
            patch_start = tuple(global_keep_start[axis] - pad[axis] for axis in range(3))
            patch_end = tuple(patch_start[axis] + image_dim[axis] for axis in range(3))
            patch = _read_reflected_box(image, patch_start, patch_end)
            if tta:
                predicted_patch = _process_with_tta(
                    patch, model, device, get_autocast(mixed_precision), low_value, high_value
                )
            else:
                predicted_patch = _process_without_tta(
                    patch, model, device, get_autocast(mixed_precision), low_value, high_value
                )
            if np.isnan(predicted_patch).any():
                warnings.warn("NaN detected in predicted patch; replacing it with background")
                predicted_patch[np.isnan(predicted_patch)] = low_value

            target_start = tuple(max(0, value) for value in patch_start)
            target_end = tuple(min(image.shape[axis], patch_end[axis]) for axis in range(3))
            target_region = tuple(slice(target_start[axis], target_end[axis]) for axis in range(3))
            source_region = tuple(
                slice(target_start[axis] - patch_start[axis], target_end[axis] - patch_start[axis])
                for axis in range(3)
            )
            local_weights = weights[source_region]
            write_zarr_region(
                sum_array, target_region,
                np.asarray(sum_array[target_region]) + predicted_patch[source_region] * local_weights,
            )
            write_zarr_region(
                weight_array, target_region,
                np.asarray(weight_array[target_region]) + local_weights,
            )
            progress.update(1)
        progress.close()

        for region in iter_chunk_slices(image.shape, output_array.chunks):
            sums = np.asarray(sum_array[region], dtype=np.float32)
            denominators = np.asarray(weight_array[region], dtype=np.float32)
            averaged = np.divide(sums, denominators, out=np.zeros_like(sums), where=denominators != 0)
            if high_value > low_value:
                averaged = (averaged - low_value) / (high_value - low_value)
            else:
                averaged.fill(0)
            write_zarr_region(output_array, region, np.clip(averaged, 0.0, 1.0))

        del group["_sum"]
        del group["_weights"]
        update_group_attributes(group, {
            "penumbria_complete": True,
            "penumbria_global_normalization": True,
        })
        completed_output = ChunkedVolume(output_path)
        outputs.append(completed_output)
        if save_files:
            export_zarr_to_tiff(
                completed_output,
                os.path.join(output_directory, f"{output_name}_inference_output.tif"),
                output_dtype=np.float32,
            )

    return outputs, output_names, [None] * len(outputs)


def _read_reflected_box(image, start, end):
    if isinstance(image, ChunkedVolume):
        return image.read_box(start, end, pad_mode="reflect")
    return read_ndarray_box(image, start, end, pad_mode="reflect")
def _process_without_tta(patch, model, device, autocast_context, low_value, high_value):
    """Run inference without test-time augmentation."""
    patch_torch = torch.from_numpy(patch).float().to(device)
    with torch.no_grad(), autocast_context:
        predicted_patch, _ = model(patch_torch.unsqueeze(0).unsqueeze(0))
    predicted_patch = predicted_patch.detach().cpu().numpy().squeeze()
    return np.clip(predicted_patch, low_value, high_value)


def _process_with_tta(patch, model, device, autocast_context, low_value, high_value):
    """Run inference with test-time augmentation (rotations)."""
    avg_patch = np.zeros_like(patch, dtype=np.float32)
    axes_list = [(1, 2), (2, 1)]
    n_transforms = 0
    tta_threshold = (high_value - low_value) * 0.01 + low_value
    break_flag = False

    for k in [0, 1, 2, 3]:
        if break_flag:
            break
        if k == 0:
            # No rotation
            prediction = _run_model(patch, model, device, autocast_context)
            some_foreground = np.any(prediction[5:-5, 5:-5, 5:-5] > tta_threshold)
            if not some_foreground:
                break_flag = True
            avg_patch += prediction
            n_transforms += 1
        else:
            # Apply rotations on different axes
            for axes in axes_list:
                rotated = np.rot90(patch, k, axes=axes).copy()
                prediction = _run_model(rotated, model, device, autocast_context)
                prediction = np.rot90(prediction, -k, axes=axes)
                avg_patch += prediction
                n_transforms += 1

    avg_patch /= n_transforms
    return np.clip(avg_patch, low_value, high_value)


def _run_model(patch, model, device, autocast_context):
    """Helper to run model inference."""
    patch_torch = torch.from_numpy(patch).float().to(device)
    with torch.no_grad(), autocast_context:
        predicted_patch, _ = model(patch_torch.unsqueeze(0).unsqueeze(0))
    return predicted_patch.detach().cpu().numpy().squeeze()
