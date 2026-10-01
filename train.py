import random
import warnings

from contextlib import nullcontext

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, minimum_filter, maximum_filter
from tqdm import tqdm

import torch
import torch.nn as nn
from U_VixLSTM.frn import GatedContextFRN3d

from utils import (
  random_rotate_batch_2d,
  generate_motion_blur_kernel,
  apply_motion_blur_kernel,
  random_rotate_and_flip_batch,
)
from volume_io import ChunkedVolume, SamplingIndex, build_sampling_index, read_ndarray_box

import torch.nn.functional as F


def _numpy_sampling_index(integer_image, image, training_image_shape):
    integer_image = np.asarray(integer_image).copy()
    integer_image[integer_image < 0] = 0
    centres = []
    for label_id, slice_tuple in enumerate(find_objects(integer_image), start=1):
        if slice_tuple is None:
            continue
        local_locs = np.asarray(np.where(integer_image[slice_tuple] == label_id))
        global_locs = np.stack(local_locs).T + np.asarray([sl.start for sl in slice_tuple])
        centres.append(np.median(global_locs, axis=0).astype(np.int64))

    image_smoothed = gaussian_filter(np.asarray(image), sigma=2)
    radius = int(training_image_shape[0]) // 2
    neighborhood = 2 * radius + 1
    background_mask = integer_image == 0
    difficulty_map = image_smoothed * background_mask
    local_max = (difficulty_map == maximum_filter(difficulty_map, size=neighborhood)) & (difficulty_map > 0)
    difficult_background = np.argwhere(local_max)

    foreground_mask = integer_image > 0
    masked_image = np.where(foreground_mask, image_smoothed, np.inf)
    local_min = (masked_image == minimum_filter(masked_image, size=neighborhood)) & foreground_mask
    difficult_foreground = np.argwhere(local_min)
    empty = np.empty((0, 3), dtype=np.int64)
    return SamplingIndex(
        np.asarray(centres, dtype=np.int64) if centres else empty,
        np.asarray(difficult_background, dtype=np.int64),
        np.asarray(difficult_foreground, dtype=np.int64),
    )


def _make_sampling_index(integer_image, image, training_image_shape):
    if isinstance(integer_image, ChunkedVolume) and isinstance(image, ChunkedVolume):
        return build_sampling_index(integer_image, image, training_image_shape)
    return _numpy_sampling_index(integer_image, image, training_image_shape)


def _read_patch(volume, centre, patch_shape, *, label_kind):
    if isinstance(volume, ChunkedVolume):
        if label_kind == "integer":
            return volume.read_patch(centre, patch_shape, pad_mode="constant", constant_values=-100)
        return volume.read_patch(centre, patch_shape, pad_mode="reflect")

    patch_shape = tuple(int(v) for v in patch_shape)
    start = tuple(int(centre[axis]) - patch_shape[axis] // 2 for axis in range(3))
    end = tuple(start[axis] + patch_shape[axis] for axis in range(3))
    if label_kind == "integer":
        return read_ndarray_box(volume, start, end, pad_mode="constant", constant_values=-100)
    return read_ndarray_box(volume, start, end, pad_mode="reflect")


def _choose_location(index, shape, strategy, jitter_radius, bounds=None):
    if strategy < 5 and len(index.object_centres):
        locations = index.object_centres
    elif strategy >= 8:
        locations = index.difficult_background if np.random.randint(0, 2) == 0 else index.difficult_foreground
    else:
        locations = None

    lower = np.zeros(3, dtype=np.int64) if bounds is None else np.asarray(bounds[0], dtype=np.int64)
    upper = np.asarray(shape, dtype=np.int64) if bounds is None else np.asarray(bounds[1], dtype=np.int64)
    upper = np.maximum(upper, lower + 1)
    if locations is not None and len(locations):
        in_bounds = np.all((locations >= lower) & (locations < upper), axis=1)
        locations = locations[in_bounds]

    if locations is None or len(locations) == 0:
        location = np.asarray([
            random.randrange(int(lower[axis]), int(upper[axis])) for axis in range(3)
        ], dtype=np.int64)
    else:
        location = np.asarray(locations[np.random.randint(0, len(locations))], dtype=np.int64).copy()
        location += np.asarray([
            np.random.randint(-radius, radius + 1) for radius in jitter_radius
        ], dtype=np.int64)
    return np.clip(location, lower, upper - 1)


class _TrainingBlockCache:
    """Keep several hundred MiB of one training triplet hot for repeated crops."""

    def __init__(self, image, heat, integer, patch_shape, anchor):
        self.sources = (image, heat, integer)
        self.patch_shape = np.asarray(patch_shape, dtype=np.int64)
        shape = np.asarray(image.shape, dtype=np.int64)
        desired_shape = np.minimum(shape, self.patch_shape * 3)
        maximum_start = shape - desired_shape
        self.start = np.minimum(np.maximum(0, np.asarray(anchor) - desired_shape // 2), maximum_start)
        self.end = self.start + desired_shape
        region = tuple(slice(int(self.start[a]), int(self.end[a])) for a in range(3))
        self.blocks = tuple(
            source.read_region(region) if isinstance(source, ChunkedVolume) else np.asarray(source)[region]
            for source in self.sources
        )

    @property
    def centre_bounds(self):
        half = self.patch_shape // 2
        lower = self.start + half
        upper = self.end - (self.patch_shape - half) + 1
        if np.any(upper <= lower):
            return self.start, self.end
        return lower, upper

    def patches(self, centre):
        centre = np.asarray(centre, dtype=np.int64)
        patch_start = centre - self.patch_shape // 2
        patch_end = patch_start + self.patch_shape
        if np.all(patch_start >= self.start) and np.all(patch_end <= self.end):
            region = tuple(
                slice(int(patch_start[a] - self.start[a]), int(patch_end[a] - self.start[a]))
                for a in range(3)
            )
            return tuple(np.ascontiguousarray(block[region]) for block in self.blocks)
        return (
            _read_patch(self.sources[0], centre, self.patch_shape, label_kind="image"),
            _read_patch(self.sources[1], centre, self.patch_shape, label_kind="heat"),
            _read_patch(self.sources[2], centre, self.patch_shape, label_kind="integer"),
        )

def get_autocast(mixed_precision=True):
    if not mixed_precision:
        return nullcontext()

    # New API (PyTorch 2.0+)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)

    # Old API (PyTorch ≤1.13)
    return torch.cuda.amp.autocast(dtype=torch.float16)

def get_grad_scaler(mixed_precision=True, device="cuda"):
    """
    Returns a GradScaler if mixed_precision is True, otherwise None.
    Compatible with different PyTorch versions.
    """
    if not mixed_precision:
        return None

    try:
        # PyTorch >=2.0 preferred import
        from torch.amp import GradScaler
    except ImportError:
        # Fallback for older PyTorch versions
        from torch.cuda.amp import GradScaler

    return GradScaler(enabled=(device == "cuda"))


def motion_blur_augmentation(input_images, device, dim=3):
    """
    Applies motion blur augmentation with probability 0.2 to each image.
    Nothing special about it, maybe a normal blur also works.
    Args:
        input_images: torch.Tensor of shape (B, H, W) for 2D or (B, D, H, W) for 3D
        device: torch.device
        dim: 2 or 3, depending on whether input is 2D or 3D

    Returns:
        Tuple of (augmented images, list of (kernel, padding) used per image) or original images
    """
    gaussprob0 = np.random.uniform(0, 1)
    if gaussprob0 > 0.8:
        kernel_size = (3,) * dim
        kernel_padding_list = []
        kernel_convolved_images = []
        for i in range(len(input_images)):
            angle = np.random.randint(1, 60)
            kernel, padding = generate_motion_blur_kernel(device, angle, kernel_size, dim)
            kernel_padding_list.append((kernel, padding))
            convolved_image = apply_motion_blur_kernel(input_images[i], kernel, padding, dim)
            kernel_convolved_images.append(convolved_image)
        input_images = torch.stack(kernel_convolved_images)
        return input_images, kernel_padding_list
    else:
        return input_images, None

def gaussian_noise_augmentation(input_images):
    gaussprob = np.random.uniform(0, 1)
    if gaussprob > 0.8:
        gaussian_tensors = torch.stack([torch.normal(mean=0, std=0.3, size=imagetensor.shape).to(input_images.device) \
                                        for imagetensor in input_images])
        input_images = torch.clamp(input_images + gaussian_tensors, min=0.0, max=1.0)
        return input_images, gaussian_tensors
    else:
        return input_images, None

def check_conv_type(model):
    has_2d = any(isinstance(layer, nn.Conv2d) for layer in model.modules())
    has_3d = any(isinstance(layer, nn.Conv3d) for layer in model.modules())
    
    if has_2d and has_3d:
        print("Model uses both 2D and 3D convolutions.")
        return -1
    elif has_2d:
        print("Model uses 2D convolutions.")
        return 0
    elif has_3d:
        print("Model uses 3D convolutions.")
        return 1
    else:
        print("Model uses neither 2D nor 3D convolutions.")
        return -1

def train_model(model, 
                optimizer, 
                loss_fn, 
                images,
                val_images, 
                labels_intensity, 
                val_labels_intensity,
                train_labels_integer,
                val_labels_integer,
                early_stopping_patience,
                device,
                pad_length,
                mixed_precision = True,
                ignore_index = -100,
                dynamic_cropping = True,
                training_image_shape = [64, 64, 64],
                keep_size = [32, 32, 32],
                mini_batch_size = 1,
                verbose = True,
                training_iterations = 100000,
                data_augmentation_types = ["rotate", 
                                           "motion_blur",
                                           "gaussian_noise"],
                print_grad_norms = False,
                evaluation_interval = 20,
                use_cell_hint = True):
    
    if mixed_precision:
        scaler = get_grad_scaler(mixed_precision)

    print("obtaining sampling locations for training and validation...")

    autocast_context = get_autocast(mixed_precision)
    if dynamic_cropping:
        train_sampling_indices = [
            _make_sampling_index(integer_image, image, training_image_shape)
            for integer_image, image in zip(train_labels_integer, images)
        ]
        val_sampling_indices = [
            _make_sampling_index(integer_image, image, training_image_shape)
            for integer_image, image in zip(val_labels_integer, val_images)
        ]
            
    validation_start = -1
    print("training started ...")
    num_mini = mini_batch_size
    checkpoint_10 = int(training_iterations * 0.1)
    checkpoint_25 = int(training_iterations * 0.25)
    checkpoint_50 = int(training_iterations * 0.5)
    checkpoint_75 = int(training_iterations * 0.75)
    checkpoint_10_saved = False
    checkpoint_25_saved = False
    checkpoint_50_saved = False
    checkpoint_75_saved = False
    best_val_loss = 1000000.0
    patience_counter = 0
    best_model = model
    train_losses = []
    val_losses = []
    num_images = len(images)
    conv_type = check_conv_type(model)
    training_shape = training_image_shape
    training_shape_half = [dim // 2 for dim in training_shape]
    if num_mini > num_images:
        warnings.warn("mini_batch_size is greater than number of training images. Setting mini_batch_size to training image length.")
        num_mini = num_images 
    print("early_stopping_patience", early_stopping_patience)
    print("data_augmentation_types", data_augmentation_types)
    active_cache = None
    active_image_index = None
    cache_reuse_remaining = 0
    cache_reuse_steps = 24

    for training_iter in tqdm(range(training_iterations)):

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        model.train()
        optimizer.zero_grad()
        if dynamic_cropping and num_mini == 1 and cache_reuse_remaining > 0:
            selected_indices = np.asarray([active_image_index], dtype=np.int64)
        else:
            indices = np.random.permutation(num_images)
            selected_indices = indices[:num_mini]
        input_images__ = [images[i] for i in selected_indices]
        labels_inten__ = [labels_intensity[i] for i in selected_indices]
        labels_integer__ = [train_labels_integer[i] for i in selected_indices]

        # dynamic cropping is used when the full image does not fit on the GPU
        if dynamic_cropping:
            if len(training_image_shape) != 3:
                raise ValueError("The out-of-core training path supports 3-D data only")
            input_images, labels_inten, labels_integer = [], [], []
            jitter_radius_list = [max(half // 4, 2) for half in training_shape_half]
            for batch_index, image_index in enumerate(selected_indices):
                image_index = int(image_index)
                new_cache = False
                if num_mini == 1 and cache_reuse_remaining <= 0:
                    anchor = _choose_location(
                        train_sampling_indices[image_index],
                        labels_integer__[batch_index].shape,
                        np.random.randint(0, 10),
                        jitter_radius_list,
                    )
                    active_cache = _TrainingBlockCache(
                        input_images__[batch_index], labels_inten__[batch_index], labels_integer__[batch_index],
                        training_image_shape, anchor,
                    )
                    active_image_index = image_index
                    cache_reuse_remaining = cache_reuse_steps
                    new_cache = True
                location_bounds = active_cache.centre_bounds if num_mini == 1 else None
                for attempt in range(100):
                    if new_cache and attempt == 0:
                        chosen_loc = anchor
                    else:
                        chosen_loc = _choose_location(
                            train_sampling_indices[image_index],
                            labels_integer__[batch_index].shape,
                            np.random.randint(0, 10),
                            jitter_radius_list,
                            bounds=location_bounds,
                        )
                    if num_mini == 1:
                        image_patch, heat_patch, integer_patch = active_cache.patches(chosen_loc)
                    else:
                        image_patch = _read_patch(
                            input_images__[batch_index], chosen_loc, training_image_shape, label_kind="image"
                        )
                        heat_patch = _read_patch(
                            labels_inten__[batch_index], chosen_loc, training_image_shape, label_kind="heat"
                        )
                        integer_patch = _read_patch(
                            labels_integer__[batch_index], chosen_loc, training_image_shape, label_kind="integer"
                        )
                    if image_patch.shape == tuple(training_image_shape) and image_patch.min() != image_patch.max():
                        break
                else:
                    raise RuntimeError(f"Could not find a non-constant training patch in image {image_index}")
                input_images.append(image_patch)
                labels_inten.append(heat_patch)
                labels_integer.append(integer_patch)
            if num_mini == 1:
                cache_reuse_remaining -= 1
        
        else:
            input_images = [np.asarray(volume[:]) if isinstance(volume, ChunkedVolume) else volume for volume in input_images__]
            labels_inten = [np.asarray(volume[:]) if isinstance(volume, ChunkedVolume) else volume for volume in labels_inten__]
            labels_integer = [np.asarray(volume[:]) if isinstance(volume, ChunkedVolume) else volume for volume in labels_integer__]

        input_images = [torch.from_numpy(arr) for arr in input_images]
        labels_inten = [torch.from_numpy(arr) for arr in labels_inten]
        labels_integer = [torch.from_numpy(arr) for arr in labels_integer]

        input_images = torch.stack(input_images).float().to(device)
        labels_inten = torch.stack(labels_inten).float().to(device)
        labels_integer = torch.stack(labels_integer).float().to(device)

        # Augmentation
        if "rotate" in data_augmentation_types:
            if len(training_image_shape) == 3:
                input_images, labels_inten, labels_integer, angles, axes = random_rotate_and_flip_batch(input_images, 
                                                                                                        labels_inten, 
                                                                                                        labels_integer)
            elif len(training_image_shape) == 2:
                input_images, labels_inten, labels_integer, angles = random_rotate_batch_2d(input_images, 
                                                                                            labels_inten, 
                                                                                            labels_integer)
                axes = None
                
        if "motion_blur" in data_augmentation_types:
            input_images, kernel_padding_list = motion_blur_augmentation(input_images, device, len(training_image_shape))
        if "gaussian_noise" in data_augmentation_types:
            input_images, gaussian_tensors = gaussian_noise_augmentation(input_images)

        if use_cell_hint:
            unqs = torch.unique(labels_integer[labels_integer>0])
            sampled_cue = False
            if len(unqs) > 1:
                sampled_id = random.sample(list(unqs), 1)[0]
                if np.random.uniform(0, 1) > 0.8:
                    sampled_cue = True
                    cue_labels_inten = torch.where(labels_integer == sampled_id, labels_inten, torch.ones_like(labels_inten)*(-5)).to(device)
                else:
                    sampled_cue = False
        else:
            sampled_cue = False

        # Forward pass
        if device.type == 'cuda':
            with autocast_context:
                if conv_type == 1:
                    input_images = input_images.unsqueeze(1)
                elif conv_type == 0:
                    input_images = input_images.squeeze()
                    input_images = input_images.unsqueeze(0).unsqueeze(0)
                else:
                    input_images = input_images.squeeze().unsqueeze(0).unsqueeze(0)
                if sampled_cue:
                    output, _ = model(input_images, cue_labels_inten.unsqueeze(0))
                else:
                    output, _ = model(input_images)
                output = output.squeeze()
                mask = torch.where(labels_inten != ignore_index, 1.0, 0.0)
                mask2 = torch.where(labels_integer != ignore_index, 1.0, 0.0)
                mask = mask * mask2
                mask = mask.bool().to(device)
                loss2 = loss_fn(output, labels_inten.squeeze()) * mask
                loss2 = torch.mean(loss2)
                loss = loss2
        else:
            output, _ = model(input_images.unsqueeze(1)) 
            output = output.squeeze(0)
            mask = torch.where(labels_inten != ignore_index, 1.0, 0.0)
            mask2 = torch.where(labels_integer != ignore_index, 1.0, 0.0)
            mask = mask * mask2
            mask = mask.bool().to(device)
            loss1 = loss_fn(output, labels_inten.squeeze()) * mask
            loss1 = torch.mean(loss1)
            loss = loss1

        train_losses.append(loss.item())

        # Backward pass
        if device.type == 'cuda':
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (training_iter+1) % evaluation_interval == 0 and (training_iter+1) > validation_start:
            if print_grad_norms:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm().item():.4e}")
            model.eval()
            loss_sum = []
            val_img_num = 0
            for val_img_large, val_inten_large, val_int_large in zip(val_images, val_labels_intensity, 
                                                                     val_labels_integer):
                if dynamic_cropping:
                    eval_locs = val_sampling_indices[val_img_num].object_centres
                    if len(eval_locs) == 0:
                        eval_locs = [np.asarray(val_img_large.shape) // 2]
                    val_img_num += 1
                    for chosen_loc in eval_locs:
                        val_img = _read_patch(val_img_large, chosen_loc, training_image_shape, label_kind="image")
                        val_inten = _read_patch(val_inten_large, chosen_loc, training_image_shape, label_kind="heat")
                        val_int = _read_patch(val_int_large, chosen_loc, training_image_shape, label_kind="integer")
                        
                        val_images_torch = torch.from_numpy(val_img).float().to(device)
                        val_integers_torch = torch.from_numpy(val_int).float().to(device)
                        val_labels_intensity_torch = torch.from_numpy(val_inten).float().to(device)

                        if device.type == 'cuda':
                            with torch.no_grad(), autocast_context:
                                if conv_type == 1:
                                    val_images_torch = val_images_torch.unsqueeze(0).unsqueeze(0)
                                elif conv_type == 0:
                                    val_images_torch = val_images_torch.squeeze()
                                    val_images_torch = val_images_torch.unsqueeze(0).unsqueeze(0)
                                else:
                                    val_images_torch = val_images_torch.squeeze().unsqueeze(0).unsqueeze(0)
                                val_output, _ = model(val_images_torch)
                                val_output = val_output.squeeze()
                                val_integers_torch = val_integers_torch.squeeze()
                                val_labels_intensity_torch = val_labels_intensity_torch.squeeze()
                                mask = torch.where(val_integers_torch != ignore_index, 1.0, 0.0)
                                second_mask = torch.where(val_labels_intensity_torch != ignore_index, 1.0, 0.0)
                                mask = mask * second_mask
                                mask = mask.bool().to(device)
                                val_loss2 = loss_fn(val_output, val_labels_intensity_torch)
                                val_loss2 = val_loss2 * mask
                                val_loss2 = torch.mean(val_loss2)
                                val_loss = val_loss2 
                                loss_sum.append(val_loss.item())
                        else:
                            with torch.no_grad():
                                val_output = model(val_images_torch.unsqueeze(1)) 
                                val_output = val_output.squeeze(1)
                                mask = torch.where(val_labels_intensity_torch != ignore_index, 1.0, 0.0)
                                val_loss1 = loss_fn(val_output, val_labels_intensity_torch) * mask
                                val_loss1 = torch.mean(val_loss1)
                                val_loss = val_loss1 
                                loss_sum.append(val_loss.item())
                else:
                    val_img_large = np.asarray(val_img_large[:]) if isinstance(val_img_large, ChunkedVolume) else val_img_large
                    val_int_large = np.asarray(val_int_large[:]) if isinstance(val_int_large, ChunkedVolume) else val_int_large
                    val_inten_large = np.asarray(val_inten_large[:]) if isinstance(val_inten_large, ChunkedVolume) else val_inten_large
                    val_images_torch = torch.from_numpy(val_img_large).float().to(device)
                    val_integers_torch = torch.from_numpy(val_int_large).float().to(device)
                    val_labels_intensity_torch = torch.from_numpy(val_inten_large).float().to(device)

                    if device.type == 'cuda':
                        with torch.no_grad(), autocast_context:
                            if conv_type == 1:
                                val_images_torch = val_images_torch.unsqueeze(0).unsqueeze(0)
                            elif conv_type == 0:
                                val_images_torch = val_images_torch
                            else:
                                val_images_torch = val_images_torch.squeeze().unsqueeze(0).unsqueeze(0)
                            val_output, _ = model(val_images_torch)
                            val_output = val_output.squeeze()
                            val_integers_torch = val_integers_torch.squeeze()
                            val_labels_intensity_torch = val_labels_intensity_torch.squeeze()
                            mask = torch.where(val_integers_torch != ignore_index, 1.0, 0.0)
                            second_mask = torch.where(val_labels_intensity_torch != ignore_index, 1.0, 0.0)
                            mask = mask * second_mask
                            mask = mask.bool().to(device)
                            val_loss1 = loss_fn(val_output, val_labels_intensity_torch) * mask
                            val_loss1 = torch.mean(val_loss1)
                            val_loss = val_loss1
                            loss_sum.append(val_loss.item())
                    else:
                        with torch.no_grad():
                            val_output = model(val_images_torch.unsqueeze(1)) 
                            val_output = val_output.squeeze(1)
                            mask = torch.where(val_labels_intensity_torch != ignore_index, 1.0, 0.0)
                            val_loss1 = loss_fn(val_output, val_labels_intensity_torch) * mask
                            val_loss1 = torch.mean(val_loss1)
                            val_loss = val_loss1 
                            loss_sum.append(val_loss.item())


            val_losses.append(np.mean(loss_sum))
            if np.mean(loss_sum) < best_val_loss:
                best_val_loss = np.mean(loss_sum)
                patience_counter = 0
                best_model = model
            else:
                patience_counter += evaluation_interval
            if training_iter >= checkpoint_10 and checkpoint_10_saved == False:
                checkpoint_10_saved = True
                torch.save(best_model.state_dict(), "checkpoint_10.pth")
            if training_iter >= checkpoint_25 and checkpoint_25_saved == False:
                checkpoint_25_saved = True
                torch.save(best_model.state_dict(), "checkpoint_25.pth")
            elif training_iter >= checkpoint_50 and checkpoint_50_saved == False:
                checkpoint_50_saved = True
                torch.save(best_model.state_dict(), "checkpoint_50.pth")
            elif training_iter >= checkpoint_75 and checkpoint_75_saved == False:
                checkpoint_75_saved = True
                torch.save(best_model.state_dict(), "checkpoint_75.pth")
            if device == 'cuda':
                torch.cuda.empty_cache()

        if (training_iter+1) % evaluation_interval == 0 and verbose and (training_iter+1) > validation_start:
            print("Minibatchnumber {}, Train Loss: {:.4f}, Val Loss: {:.4f}, patience_counter: {}"\
                .format(training_iter, loss.item(), np.mean(loss_sum), patience_counter)) 
                
    with open("val_losses.txt", "w") as file:
        for item in val_losses:
            file.write(f"{item}\n")

    with open("train_losses.txt", "w") as file:
        for item in train_losses:
            file.write(f"{item}\n")

    return best_model
