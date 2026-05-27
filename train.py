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

import torch.nn.functional as F

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
                use_sgg_layer = True):
    
    if mixed_precision:
        scaler = get_grad_scaler(mixed_precision)

    print("obtaining sampling locations for training and validation...")

    autocast_context = get_autocast(mixed_precision)
    possible_centre_locations = {}
    total_training_objects = 0

    for idx, integer_image in enumerate(train_labels_integer):
        locs = np.argwhere(integer_image > 0)
        possible_centre_locations[idx] = locs
        total_training_objects += integer_image.max()

    if dynamic_cropping:

        possible_centre_locations_val = {}
        for idx, integer_image in enumerate(val_labels_integer):
            integer_image[integer_image < 0] = 0
            slices = find_objects(integer_image)
            for i, slice_tuple in enumerate(slices, start=1):
                if slice_tuple is not None:
                    local_locs = np.array(np.where(integer_image[slice_tuple] == i))
                    global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
                    median_loc = np.median(global_locs, axis = 0).astype(int)
                    if possible_centre_locations_val.get(idx) is None:
                        possible_centre_locations_val[idx] = [median_loc]
                    else:
                        possible_centre_locations_val[idx].append(median_loc)

        possible_centre_locations_train = {}
        for idx, integer_image in enumerate(train_labels_integer):
            integer_image[integer_image < 0] = 0
            slices = find_objects(integer_image)
            for i, slice_tuple in enumerate(slices, start=1):
                if slice_tuple is not None:
                    local_locs = np.array(np.where(integer_image[slice_tuple] == i))
                    global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
                    median_loc = np.median(global_locs, axis=0).astype(int)
                    if possible_centre_locations_train.get(idx) is None:
                        possible_centre_locations_train[idx] = [median_loc]
                    else:
                        possible_centre_locations_train[idx].append(median_loc)

        # add background elements to training locations
        difficult_background_locations_train = {}
        for idx, (integer_image, image) in enumerate(zip(train_labels_integer, images)):
            background_mask  = integer_image == 0
            image_smoothed = gaussian_filter(image, sigma=2)
            difficulty_map = image_smoothed * background_mask
            nms_radius = training_image_shape[0] // 2
            neighborhood = (2*nms_radius + 1)
            local_max = (difficulty_map == maximum_filter(difficulty_map, size=neighborhood))
            local_max &= (difficulty_map > 0)
            coords = np.argwhere(local_max)
            difficult_background_locations_train[idx] = coords

        
        difficult_foreground_locations_train = {}
        for idx, (integer_image, image) in enumerate(zip(train_labels_integer, images)):
            foreground_mask = integer_image > 0
            image_smoothed = gaussian_filter(image, sigma=2)
            masked_image = np.where(foreground_mask, image_smoothed, np.inf)
            nms_radius = training_image_shape[0] // 2
            neighborhood = 2 * nms_radius + 1
            local_min = (masked_image == minimum_filter(masked_image, size=neighborhood))
            local_min &= foreground_mask  # keep only foreground points
            coords = np.argwhere(local_min)
            difficult_foreground_locations_train[idx] = coords
            
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

    for training_iter in tqdm(range(training_iterations)):

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        model.train()
        optimizer.zero_grad()
        indices = np.random.permutation(num_images)  
        selected_indices = indices[:num_mini]  
        input_images__ = [images[i] for i in selected_indices]
        labels_inten__ = [labels_intensity[i] for i in selected_indices]
        labels_integer__ = [train_labels_integer[i] for i in selected_indices]

        # dynamic cropping is used when the full image does not fit on the GPU
        if dynamic_cropping:
            max_min_same = True
            while max_min_same:
                if len(training_image_shape) == 3:
                    Z, Y, X = labels_integer__[0].shape
                    sampled_strategy = np.random.randint(0, 10)
                    if sampled_strategy < 5:
                        jitter_radius_list = [max(kk // 4, 2) for kk in training_shape_half]
                        possible_centre_locations_train_selected = possible_centre_locations_train[\
                            selected_indices.item()]
                        chosen_loc_index = np.random.randint(0, len(possible_centre_locations_train_selected))
                        chosen_loc = possible_centre_locations_train_selected[chosen_loc_index]
                        chosen_loc = [loc + np.random.randint(-jitter_radius_list[jjj], jitter_radius_list[jjj]+1)\
                                       for jjj, loc in enumerate(chosen_loc)]
                    else:
                        if sampled_strategy >= 8:
                            second_sampler = np.random.randint(0, 2)
                            if second_sampler == 0:
                                possible_difficult_bg_locs = difficult_background_locations_train.get(\
                                    selected_indices.item())
                            else:
                                possible_difficult_bg_locs = difficult_foreground_locations_train.get(\
                                    selected_indices.item())
                            if possible_difficult_bg_locs is None:
                                chosen_loc = (random.randrange(pad_length, Z-pad_length), 
                                              random.randrange(pad_length, Y-pad_length), 
                                              random.randrange(pad_length, X-pad_length))
                            else:
                                if possible_difficult_bg_locs.shape[0] == 0:
                                    chosen_loc = (random.randrange(pad_length, Z-pad_length), 
                                                  random.randrange(pad_length, Y-pad_length), 
                                                  random.randrange(pad_length, X-pad_length))
                                else:
                                    chosen_loc_index = np.random.randint(0, len(possible_difficult_bg_locs))
                                    chosen_loc = possible_difficult_bg_locs[chosen_loc_index]
                                    jitter_radius_list = [max(kk // 4, 2) for kk in training_shape_half]
                                    chosen_loc = [loc + np.random.randint(-jitter_radius_list[jjj], 
                                                                          jitter_radius_list[jjj]+1) \
                                                                            for jjj, loc in enumerate(chosen_loc)]
                        else:
                            chosen_loc = (random.randrange(pad_length, Z-pad_length), 
                                          random.randrange(pad_length, Y-pad_length), 
                                          random.randrange(pad_length, X-pad_length))
                        

                    input_images = [input_images__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                      + training_shape_half[0], 
                                                      chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                      + training_shape_half[1], 
                                                      chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                      + training_shape_half[2]] for i in range(num_mini)]
                    labels_inten = [labels_inten__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                      + training_shape_half[0], 
                                                      chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                      + training_shape_half[1], 
                                                      chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                      + training_shape_half[2]] for i in range(num_mini)]
                    labels_integer = [labels_integer__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                          + training_shape_half[0], 
                                                          chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                          + training_shape_half[1], 
                                                          chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                          + training_shape_half[2]] for i in range(num_mini)]
                    try:
                        if input_images[0].min() != input_images[0].max():
                            if input_images[0].shape[0] == training_image_shape[0] and \
                                input_images[0].shape[1] == training_image_shape[1] and \
                                input_images[0].shape[2] == training_image_shape[2]:
                                max_min_same = False
                            else:
                                # don't use uneven cubes
                                max_min_same = True
                        else:
                            # don't sample complete background
                            max_min_same = True
                    except:
                        max_min_same = True
                elif len(training_image_shape) == 2:
                    Y, X = labels_integer__[0].shape
                    chosen_loc = (random.randrange(pad_length, Y-pad_length), random.randrange(pad_length, X-pad_length))
                    input_images = [input_images__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                      + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1]] for i in range(num_mini)]
                    labels_inten = [labels_inten__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                      + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1]] for i in range(num_mini)]
                    labels_integer = [labels_integer__[i][chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                          + training_shape_half[0], 
                                                        chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                        + training_shape_half[1]] for i in range(num_mini)]
                    try:
                        if input_images[0].min() != input_images[0].max():
                            max_min_same = False
                    except:
                        max_min_same = True

                else:
                    raise ValueError(f"Provided incorrect training dimensionality {training_image_shape}")
        
        else:
            input_images = input_images__
            labels_inten = labels_inten__
            labels_integer = labels_integer__

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

        if use_sgg_layer:
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
                    eval_locs = possible_centre_locations_val.get(val_img_num)
                    val_img_num += 1
                    for chosen_loc in eval_locs:
                        if len(training_image_shape) == 3:
                            val_img = val_img_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                    + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1], 
                                                    chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                    + training_shape_half[2]]
                            val_inten = val_inten_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                        + training_shape_half[0], 
                                                        chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                        + training_shape_half[1], 
                                                        chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                        + training_shape_half[2]]
                            val_int = val_int_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                    + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1], 
                                                    chosen_loc[2] - training_shape_half[2]:chosen_loc[2] \
                                                    + training_shape_half[2]]
                        elif len(training_image_shape) == 2:
                            val_img = val_img_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                    + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1]]
                            val_inten = val_inten_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                        + training_shape_half[0], 
                                                        chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                        + training_shape_half[1]]
                            val_int = val_int_large[chosen_loc[0] - training_shape_half[0]:chosen_loc[0] \
                                                    + training_shape_half[0], 
                                                    chosen_loc[1] - training_shape_half[1]:chosen_loc[1] \
                                                    + training_shape_half[1]]
                        else:
                            raise ValueError(f"Provided incorrect training dimensionality {training_image_shape}")
                        
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
