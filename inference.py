import torch
import numpy as np
from tqdm import tqdm
import tifffile
import os
import warnings
from contextlib import nullcontext
from utils import preprocess_0_1
from scipy.ndimage import distance_transform_edt


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

def sliding_window_loop_anisotropic(
    D, H, W, keep_size, step_size, image_dim, model,
    low_value, high_value, mixed_precision, image, 
    device, tta=False, offset=True, patch_based_norm=False, euclidean_feathering=False):

    autocast_context = get_autocast(mixed_precision)
    
    # Calculate padding for each dimension
    pad_d = (image_dim[0] - keep_size[0]) // 2
    pad_h = (image_dim[1] - keep_size[1]) // 2
    pad_w = (image_dim[2] - keep_size[2]) // 2

    # Initialize output volumes
    output_volume = np.zeros((D, H, W), dtype=np.float32)
    
    if euclidean_feathering:
        total_weights = np.zeros_like(output_volume, dtype=np.float32)
        weight_times_value = np.zeros_like(output_volume, dtype=np.float32)
        edt_array = center_weighted_array((image_dim[0], image_dim[1], image_dim[2]))
    else:
        count_volume = np.zeros_like(output_volume, dtype=np.float32)

    # Calculate offsets
    step_d, step_h, step_w = step_size
    offset_d = step_d // 2 if offset else 0
    offset_h = step_h // 2 if offset else 0
    offset_w = step_w // 2 if offset else 0

    # Sliding window loop
    for d in tqdm(range(offset_d, D - keep_size[0] + 1, step_d)):
        for h in range(offset_h, H - keep_size[1] + 1, step_h):
            for w in range(offset_w, W - keep_size[2] + 1, step_w):
                # Extract and process patch
                patch, d_start, h_start, w_start = _extract_patch(
                    image, d, h, w, pad_d, pad_h, pad_w, image_dim
                )
                
                # Skip if patch is too small
                if patch is None:
                    continue
                
                # Preprocess if needed
                if patch_based_norm:
                    patch = preprocess_0_1(patch, low_clip=1.0, high_clip=99.9)
                
                # Run inference
                if tta:
                    predicted_patch = _process_with_tta(patch, model, device, autocast_context, low_value, high_value)
                else:
                    predicted_patch = _process_without_tta(patch, model, device, autocast_context, low_value, high_value)
                
                # Handle NaN values
                if np.isnan(predicted_patch).any():
                    warnings.warn("NaN detected in predicted_patch, subbing with background")
                    predicted_patch[np.isnan(predicted_patch)] = low_value

                # Accumulate results
                if euclidean_feathering:
                    _accumulate_with_feathering(
                        weight_times_value, total_weights, predicted_patch, 
                        edt_array, d_start, h_start, w_start, image_dim
                    )
                else:
                    _accumulate_center_region(
                        output_volume, count_volume, predicted_patch,
                        d, h, w, d_start, h_start, w_start, keep_size
                    )
    
    # Normalize output
    if euclidean_feathering:
        output_volume = np.divide(weight_times_value, total_weights, 
                                 out=np.zeros_like(output_volume), 
                                 where=total_weights!=0)
    else:
        output_volume = np.divide(output_volume, count_volume, 
                                 out=np.zeros_like(output_volume), 
                                 where=count_volume!=0)
    
    return output_volume


def _extract_patch(image, d, h, w, pad_d, pad_h, pad_w, image_dim):
    """Extract a patch from the image with proper boundary handling."""
    # Calculate patch boundaries
    d_start = max(0, d - pad_d)
    h_start = max(0, h - pad_h)
    w_start = max(0, w - pad_w)

    d_end = min(image.shape[0], d_start + image_dim[0])
    h_end = min(image.shape[1], h_start + image_dim[1])
    w_end = min(image.shape[2], w_start + image_dim[2])

    # Check if patch would be too small
    if (d_end - d_start < image_dim[0] or
        h_end - h_start < image_dim[1] or
        w_end - w_start < image_dim[2]):
        return None, None, None, None

    patch = image[d_start:d_end, h_start:h_end, w_start:w_end]
    return patch, d_start, h_start, w_start


def _accumulate_with_feathering(weight_times_value, total_weights, predicted_patch, 
                                edt_array, d_start, h_start, w_start, image_dim):
    """Accumulate predictions using euclidean distance feathering."""
    d_end = d_start + image_dim[0]
    h_end = h_start + image_dim[1]
    w_end = w_start + image_dim[2]
    
    weight_times_value[d_start:d_end, h_start:h_end, w_start:w_end] += predicted_patch * edt_array
    total_weights[d_start:d_end, h_start:h_end, w_start:w_end] += edt_array


def _accumulate_center_region(output_volume, count_volume, predicted_patch,
                              d, h, w, d_start, h_start, w_start, keep_size):
    """Accumulate predictions by extracting center region only."""
    # Calculate actual padding used (accounts for boundary cases)
    actual_pad_d = d - d_start
    actual_pad_h = h - h_start  
    actual_pad_w = w - w_start

    # Extract center region from predicted patch
    center_patch = predicted_patch[
        actual_pad_d:actual_pad_d + keep_size[0],
        actual_pad_h:actual_pad_h + keep_size[1],
        actual_pad_w:actual_pad_w + keep_size[2]
    ]

    # Add to output
    output_volume[d:d + keep_size[0],
                  h:h + keep_size[1],
                  w:w + keep_size[2]] += center_patch
    
    count_volume[d:d + keep_size[0],
                 h:h + keep_size[1],
                 w:w + keep_size[2]] += 1

def compute_padding(D, step, keep_size, input_tile):
    side_padding = 2 * (input_tile - keep_size)
    leftover = D % step
    return leftover + side_padding

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
    
    # During inference, we move over the image in steps of step_size, doing inference on images
    # of shape training_shape, and keep the middle part of the image denoted by keep_size if we are not feathering

    # For feathering, we keep the same sliding window moving scheme, but instead of
    # discarding the edges, we merge overlapping patches using euclidean-distance-based feathering

    # For example (+ are kept):

    # [ddd|++++++|ddd]   (Window 1)
    #        [ddd|++++++|ddd]   (Window 2)
    #               [ddd|++++++|ddd]   (Window 3)
    
    # Validate step_size <= keep_size
    assert step_size[0] <= keep_size[0] and step_size[1] <= keep_size[1], \
        "Step size must be less than or equal to keep size"
    if len(keep_size) == 3:
        assert step_size[2] <= keep_size[2], "Step size must be less than or equal to keep size"

    # Setup output directory
    predicted_label_path = os.path.join(predicted_label_path, "preds")
    os.makedirs(predicted_label_path, exist_ok=True)

    # Validate input format
    if not isinstance(inference_images, list):
        raise Exception("inference images is not a list \n. It should be a list, even for one image (e.g. [image])")
    
    # Prepare images with padding
    inference_images_torch = []
    padding_list = []
    
    for idx, image in enumerate(inference_images):
        # Save preprocessed input if requested
        if inference_filenames is not None and save_files:
            filepath = os.path.join(predicted_label_path, f"{inference_filenames[idx]}_input_image_preprocessed.tif")
            tifffile.imwrite(filepath, image.astype(np.float32))
        
        # Check if padding needed
        image_shape = image.shape
        needs_padding = any(image_shape[i] != image_dim[i] for i in range(3))
        
        if needs_padding:
            # Compute padding requirements
            padding_width = [image_dim[i] + compute_padding(image_shape[i] + image_dim[i], 
                                                           step_size[i], keep_size[i], image_dim[i]) 
                           for i in range(3)]
            padded_values = [(pw // 2, pw - pw // 2) for pw in padding_width]
            
            # Apply padding
            padded_image = np.pad(image, pad_width=padded_values, mode='reflect')
            padding_list.append(padded_values)
        else:
            padded_image = image
            padding_list.append(None)
        
        inference_images_torch.append(padded_image)
    
    inference_images = inference_images_torch
    
    # Print inference configuration
    print(f"{'with' if tta else 'without'} test time augmentation (TTA)")
    if patch_based_norm:
        print("with patch based norm")
        print("euclidean-distance feathering if image larger than training patches")
    else:
        print("chopping and concatenating image subcubes, no feathering")
        if tta:
            print("two inference runs per image, one shifted")
    
    # Run inference on each image
    out_ims = []
    autocast_context = get_autocast(mixed_precision)
    
    for i, image in enumerate(inference_images):
        needs_sliding_window = any(image.shape[j] != image_dim[j] for j in range(3))
        
        if needs_sliding_window:
            D, H, W = image.shape
            output_volume = sliding_window_loop_anisotropic(
                D, H, W, keep_size, step_size, image_dim, model, 
                low_value, high_value, mixed_precision, image, device, 
                tta=tta, offset=False, patch_based_norm=patch_based_norm, 
                euclidean_feathering=True
            )
        else:
            output_volume = _process_single_patch(
                image, model, device, autocast_context, 
                patch_based_norm, tta, low_value, high_value
            )
        
        # Crop padding and normalize
        output_volume = _crop_and_normalize(
            output_volume, padding_list[i], low_value, high_value
        )
        
        # Save output
        if inference_filenames is not None and save_files:
            filepath = os.path.join(predicted_label_path, f"{inference_filenames[i]}_inference_output.tif")
            tifffile.imwrite(filepath, output_volume.astype(np.float32))
        
        out_ims.append(output_volume)

    return out_ims, inference_filenames, padding_list


def _process_single_patch(patch, model, device, autocast_context, patch_based_norm, tta, low_value, high_value):
    """Process a single patch (no sliding window needed)."""
    if patch_based_norm:
        patch = preprocess_0_1(patch, low_clip=1.0, high_clip=99.9)
    
    if tta:
        return _process_with_tta(patch, model, device, autocast_context, low_value, high_value)
    else:
        return _process_without_tta(patch, model, device, autocast_context, low_value, high_value)


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
    
    for k in [0, 1, 2, 3]:
        if k == 0:
            # No rotation
            prediction = _run_model(patch, model, device, autocast_context)
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


def _crop_and_normalize(output_volume, padding, low_value, high_value):
    """Crop padding and normalize output volume."""
    # Crop padding if it was applied
    if padding is not None:
        output_shape = output_volume.shape
        output_volume = output_volume[
            padding[0][0]:output_shape[0]-padding[0][1],
            padding[1][0]:output_shape[1]-padding[1][1],
            padding[2][0]:output_shape[2]-padding[2][1]
        ]
    
    # Normalize to [0, 1]
    if output_volume.max() != output_volume.min():
        output_volume = (output_volume - low_value) / (high_value - low_value)
    else:
        output_volume = np.zeros(output_volume.shape, dtype=np.float32)
    
    return output_volume