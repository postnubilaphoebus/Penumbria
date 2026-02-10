import numpy as np
import os.path
import torch
import torch.nn.functional as F
import re
import warnings
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter
import skimage

def load_inference_images(inference_path, fileformat='.tif'):
    print("loading inference images...")
    images = []
    filenames = []
    for filename in os.listdir(inference_path):
        if filename.endswith(fileformat):
            imagepath = os.path.join(inference_path, filename)
            image = skimage.io.imread(imagepath)
            images.append(image)
            name_without_extension = os.path.splitext(filename)[0]
            filenames.append(name_without_extension)
    return images, filenames


def random_rotate_batch_2d(images, labels, integer_labels):
    batch_size, height, width = images.size()
    
    # Randomly select 0, 1, 2, or 3 quarter turns for each image
    angles = torch.randint(0, 4, (batch_size,))
    
    rotated_images = []
    rotated_labels = []
    rotated_integer_labels = []

    for i in range(batch_size):
        angle = angles[i].item()
        
        rotated_image = images[i].rot90(angle, dims=(0, 1))
        rotated_label = labels[i].rot90(angle, dims=(0, 1))
        rotated_integer_label = integer_labels[i].rot90(angle, dims=(0, 1))
        
        rotated_images.append(rotated_image)
        rotated_labels.append(rotated_label)
        rotated_integer_labels.append(rotated_integer_label)

    return (
        torch.stack(rotated_images),
        torch.stack(rotated_labels),
        torch.stack(rotated_integer_labels),
        angles
    )

# def random_rotate_batch(images, labels, integer_labels):
#     batch_size, depth, height, width = images.size()

#     axes = torch.randint(0, 3, (batch_size,))
#     angles = []

#     for axis in axes:
#         if axis == 0:
#             # Rotation in XY plane (around Z): only allow 0 or 180 degrees (0 or 2 quarter turns)
#             angle = torch.randint(0, 2, (1,)) * 2  # 0 or 2
#         else:
#             # Allow 0, 90, 180, 270 for other axes
#             angle = torch.randint(0, 4, (1,))
#         angles.append(angle.item())

#     angles = torch.tensor(angles)

#     rotated_images = []
#     rotated_labels = []
#     rotated_integer_labels = []

#     for i in range(batch_size):
#         image = images[i]
#         label = labels[i]
#         integer_label = integer_labels[i]

#         if axes[i] == 0:
#             dims = (1, 2)  # Rotate in XY plane (around Z axis)
#         elif axes[i] == 1:
#             dims = (0, 2)  # Rotate in XZ plane (around Y axis)
#         elif axes[i] == 2:
#             dims = (0, 1)  # Rotate in YZ plane (around X axis)

#         rotated_image = image.rot90(angles[i], dims=dims)
#         rotated_label = label.rot90(angles[i], dims=dims)
#         rotated_integer_label = integer_label.rot90(angles[i], dims=dims)

#         rotated_images.append(rotated_image)
#         rotated_labels.append(rotated_label)
#         rotated_integer_labels.append(rotated_integer_label)

#     return (
#         torch.stack(rotated_images),
#         torch.stack(rotated_labels),
#         torch.stack(rotated_integer_labels),
#         angles,
#         axes
    #)
import torch

def random_rotate_batch(images, labels, integer_labels):
    """
    Apply one of 10 distinct 3D rotations to a batch of 3D images.
    
    Only includes rotations that preserve the discrete grid structure:
    - 1 identity rotation (0°)
    - 3 rotations around X-axis (90°, 180°, 270°)  
    - 3 rotations around Y-axis (90°, 180°, 270°)
    - 3 rotations around Z-axis (90°, 180°, 270°)
    
    These are the only rotations that can be applied to discrete 3D grids
    without interpolation or distortion.
    """
    batch_size, depth, height, width = images.size()
    
    # Choose random rotation type for each sample in batch
    rotation_types = torch.randint(0, 10, (batch_size,))
    
    rotated_images = []
    rotated_labels = []
    rotated_integer_labels = []
    rotation_info = []
    
    for i in range(batch_size):
        image = images[i]
        label = labels[i]
        integer_label = integer_labels[i]
        rot_type = rotation_types[i].item()
        
        if rot_type == 0:
            # Identity - no rotation
            rotated_image = image
            rotated_label = label
            rotated_integer_label = integer_label
            info = ("identity", 0, None)
            
        elif rot_type in [1, 2, 3]:
            # Rotations around X-axis (90°, 180°, 270°)
            k = rot_type  # 1, 2, 3 quarter turns
            dims = (1, 2)  # YZ plane
            rotated_image = image.rot90(k, dims=dims)
            rotated_label = label.rot90(k, dims=dims)
            rotated_integer_label = integer_label.rot90(k, dims=dims)
            info = ("x_axis", k * 90, dims)
            
        elif rot_type in [4, 5, 6]:
            # Rotations around Y-axis (90°, 180°, 270°)
            k = rot_type - 3  # 1, 2, 3 quarter turns
            dims = (0, 2)  # XZ plane
            rotated_image = image.rot90(k, dims=dims)
            rotated_label = label.rot90(k, dims=dims)
            rotated_integer_label = integer_label.rot90(k, dims=dims)
            info = ("y_axis", k * 90, dims)
            
        elif rot_type in [7, 8, 9]:
            # Rotations around Z-axis (90°, 180°, 270°)
            k = rot_type - 6  # 1, 2, 3 quarter turns
            dims = (0, 1)  # XY plane
            rotated_image = image.rot90(k, dims=dims)
            rotated_label = label.rot90(k, dims=dims)
            rotated_integer_label = integer_label.rot90(k, dims=dims)
            info = ("z_axis", k * 90, dims)
            
        rotated_images.append(rotated_image)
        rotated_labels.append(rotated_label)
        rotated_integer_labels.append(rotated_integer_label)
        rotation_info.append(info)
    
    return (
        torch.stack(rotated_images),
        torch.stack(rotated_labels),
        torch.stack(rotated_integer_labels),
        rotation_types,
        rotation_info
    )

# If you need more rotations, here's how to add reflections (flips)
def random_rotate_and_flip_batch(images, labels, integer_labels):
    """
    Extended version that includes rotations + reflections for 48 total transformations.
    
    Combines the 10 rotations with flips along each axis for more data augmentation.
    All transformations preserve the discrete grid structure.
    """
    batch_size, depth, height, width = images.size()
    
    # First apply rotation
    images_rot, labels_rot, integer_labels_rot, rot_types, rot_info = random_rotate_batch(
        images, labels, integer_labels
    )
    
    # Then randomly apply flips
    flip_x = torch.rand(batch_size) > 0.5
    flip_y = torch.rand(batch_size) > 0.5  
    flip_z = torch.rand(batch_size) > 0.5
    
    for i in range(batch_size):
        if flip_x[i]:
            images_rot[i] = torch.flip(images_rot[i], dims=[0])
            labels_rot[i] = torch.flip(labels_rot[i], dims=[0])
            integer_labels_rot[i] = torch.flip(integer_labels_rot[i], dims=[0])
        if flip_y[i]:
            images_rot[i] = torch.flip(images_rot[i], dims=[1])
            labels_rot[i] = torch.flip(labels_rot[i], dims=[1])
            integer_labels_rot[i] = torch.flip(integer_labels_rot[i], dims=[1])
        if flip_z[i]:
            images_rot[i] = torch.flip(images_rot[i], dims=[2])
            labels_rot[i] = torch.flip(labels_rot[i], dims=[2])
            integer_labels_rot[i] = torch.flip(integer_labels_rot[i], dims=[2])
    
    return (
        images_rot,
        labels_rot,
        integer_labels_rot,
        rot_types,
        (flip_x, flip_y, flip_z)
    )

    
def apply_motion_blur_kernel(volume, kernel, padding, dim):
    if dim == 3:
        return F.conv3d(volume.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding).squeeze(0).squeeze(0)
    elif dim == 2:
        return F.conv2d(volume.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding).squeeze(0).squeeze(0)
    else:
        raise ValueError(f"Invalid dimension: {dim}")

def generate_motion_blur_kernel(device, angle, kernel_sizes, dim):
    angle_rad = torch.tensor(np.radians(angle), dtype=torch.float32)

    # 2D or 3D kernel
    kernel = torch.zeros(kernel_sizes, dtype=torch.float32)
    center = [(size - 1) / 2 for size in kernel_sizes]
    offsets = [torch.linspace(-c, c, steps=size, dtype=torch.float32) for c, size in zip(center, kernel_sizes)]

    if dim == 3:
        x, y, z = torch.meshgrid(*offsets, indexing='ij')
        blurring_plane = np.random.choice(3)
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        if blurring_plane == 0:
            direction = torch.tensor([cos_theta, sin_theta, 0], dtype=torch.float32)
        elif blurring_plane == 1:
            direction = torch.tensor([cos_theta, 0, sin_theta], dtype=torch.float32)
        else:
            direction = torch.tensor([0, cos_theta, sin_theta], dtype=torch.float32)

        displacement = torch.stack((direction[0] * x, direction[1] * y, direction[2] * z), dim=-1)
        kernel += torch.linalg.norm(displacement, dim=-1)

    elif dim == 2:
        x, y = torch.meshgrid(*offsets, indexing='ij')
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        # Rotate in the XY plane
        dx = cos_theta * x + sin_theta * y
        dy = -sin_theta * x + cos_theta * y
        displacement = torch.stack((dx, dy), dim=-1)
        kernel += torch.linalg.norm(displacement, dim=-1)

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    kernel /= kernel.sum()
    kernel = kernel.to(device)

    padding = [ks // 2 for ks in kernel_sizes]
    return kernel, padding

def preprocess(img, low_clip=96, high_clip=99.9, clip=False, normalize=True, moment_standardization=False, scale_factor = 2.0):
    '''
    Percentile clips image
    and normalizes to [0,N] - N/2 
    '''
    if clip:
        min_clip = np.percentile(img, low_clip)
        max_clip = np.percentile(img, high_clip)
        img = np.clip(img, min_clip, max_clip)

    if normalize:
        if moment_standardization:
            img_mean = img.mean()
            img_std = img.std()
            img_std = np.std(img)
            if img_std != 0:
                img = (img - img_mean) / img_std
            else:
                img = img - img_mean
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            warnings.warn("warning: image has only one value:" + str(img.max()))
            img = np.ones_like(img) * img.max()
        img = scale_factor * img - (scale_factor / 2)
        
    return img

def preprocess_0_1(img, low_clip=1.0, high_clip=99.0, clip=True):
    if clip:
        min_clip = np.percentile(img, low_clip)
        max_clip = np.percentile(img, high_clip)
        img = np.clip(img, min_clip, max_clip)

    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        warnings.warn("warning: image has only one value:" + str(img.max()))
        img = np.ones_like(img) * img.max()
        
    return img

def load_masks(mask_path, fileformat='.npy'):
    masks = []
    filenames = []
    for filename in os.listdir(mask_path):
        if filename.endswith(fileformat):
            if "mask" in filename:
                maskpath = os.path.join(mask_path, filename)
                mask = np.load(maskpath)
                mask_max = mask.max()
                mask_min = mask.min()
                if mask_max == mask_min:
                    warnings.warn("warning: mask has only one value:" + str(mask_min))
                    warnings.warn("offending file: " + filename)
                else:
                    mask_half = (mask_max + mask_min) / 2
                    mask = mask > mask_half
                smoothed_mask = gaussian_filter(mask.astype(np.float32), sigma=3)
                smoothed_mask = smoothed_mask > 0.4
                masks.append(smoothed_mask)
                name_without_extension = os.path.splitext(filename)[0]
                filenames.append(name_without_extension)
    return masks, filenames

def load_training_images_and_labels(training_path,
                                    image_format=".tif",
                                    label_format=".npy",
                                    label_to_int=True):
    """
    Load training images and corresponding labels from a directory.
    Args:
        training_path (str): Path to the directory containing the training files.
        num_images (int): Number of images to load.
        image_format (str): File format of the images (e.g., '.tif', '.png').
        label_format (str): File format of the labels (e.g., '.npy', '.tif').
        label_to_int (bool): Whether to convert labels to integers.
    Returns:
        images (list): List of loaded image arrays.
        labels (list): List of corresponding label arrays.
        integer_labels (list): List of corresponding integer label arrays (if found).
    """
    print("Loading train images and labels...")
    images = []
    labels = []
    integer_labels = []
    
    # List and sort files in the training path
    files = os.listdir(training_path)

    def extract_number(f):
        match = re.search(r'\d+', f)
        return int(match.group()) if match else -1

    image_files = sorted(
        [f for f in files if f.endswith(image_format)
        and "label" not in f.lower()
        and "mask" not in f.lower()
        and "integer" not in f.lower()
        and re.search(r'\d+', f)],
        key=extract_number
    )

    label_files = sorted(
        [f for f in files if f.endswith(label_format)
        and ("label" in f.lower() or "mask" in f.lower())
        and "integer" not in f.lower()
        and re.search(r'\d+', f)],
        key=extract_number
    )

    integer_label_files = sorted(
        [f for f in files if (f.endswith(".npy") or f.endswith(".tif"))
        and "integer" in f.lower()
        and re.search(r'\d+', f)],
        key=extract_number
    )

    resizing_file = [f for f in files if "resizing_factors" in f.lower()]
    resizing_file = resizing_file[0]

    with open(os.path.join(training_path, resizing_file), "r") as file:
        resizing_factors = file.readline()
        resizing_factors = resizing_factors.strip("[").strip("]")
        resizing_factors = resizing_factors.split(",")
        resizing_factors = [float(f) for f in resizing_factors]
        print("rounding resizing factors to 2 decimal places...")
        resizing_factors = [round(f, 2) for f in resizing_factors]

    if len(image_files) == 0:
        raise FileNotFoundError("No images found, check naming conventions and folder contents")
    if len(label_files) == 0:
        raise FileNotFoundError("No heat labels found, check naming conventions and folder contents")
    if len(integer_label_files) == 0:
        raise FileNotFoundError("No labels found, check naming conventions and folder contents")
    
    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        # Match numeric IDs in filenames
        image_num = re.search(r'\d+', image_file).group()
        label_num = re.search(r'\d+', label_file).group()
        
        if image_num == label_num:  # Ensure matching IDs
            imagepath = os.path.join(training_path, image_file)
            labelpath = os.path.join(training_path, label_file)
            
            # Load the image and label
            image = iio.imread(imagepath)
            
            if label_format == ".tif":
                label = iio.imread(labelpath)
            else:
                label = np.load(labelpath)
            
            if label_to_int:
                label = label.astype(int)
            
            images.append(image)
            labels.append(label)
            
            # Look for matching integer label file
            matching_integer_file = None
            for int_file in integer_label_files:
                int_num = re.search(r'\d+', int_file).group()
                if int_num == image_num:
                    matching_integer_file = int_file
                    break
            
            if matching_integer_file:
                integer_labelpath = os.path.join(training_path, matching_integer_file)
                try:
                    integer_label = np.load(integer_labelpath).astype(int)
                except:
                    integer_label = iio.imread(integer_labelpath).astype(int)
                integer_labels.append(integer_label)
            else:
                # If no matching integer label found, append None or empty array
                integer_labels.append(None)

            print("Loaded image {} and label {} and integer label {}.".format(image_file, label_file, matching_integer_file))
    
    if len(images) != len(labels) or len(images) != len(integer_labels):
        raise ValueError("Number of images and labels do not match. Consider eg the file loading format")
    
    # Return integer_labels as well
    return images, labels, integer_labels, resizing_factors

def load_both_upsampled_and_normal_labels(normal_path, upsampled_path, image_format=".tif", label_format=".tif", label_to_int=True):
    images, labels, integer_labels = load_training_images_and_labels(normal_path, 
                                                                     image_format=image_format, 
                                                                     label_format=label_format, 
                                                                     label_to_int=label_to_int)
    images2, labels2, integer_labels2 = load_training_images_and_labels(upsampled_path, 
                                                                        image_format=image_format, 
                                                                        label_format=label_format, 
                                                                        label_to_int=label_to_int)
    return images, labels, integer_labels, images2, labels2, integer_labels2


def load_training_filenames(training_path, image_format=".tif", label_format=".npy"):
    """
    Load filenames of training images and corresponding labels from a directory.
    Args:
        training_path (str): Path to the directory containing the training files.
        num_images (int): Number of images to load.
        image_format (str): File format of the images (e.g., '.tif', '.png').
        label_format (str): File format of the labels (e.g., '.npy', '.tif').
    Returns:
        image_files (list): List of image filenames.
        label_files (list): List of label filenames.
        integer_label_files (list): List of integer label filenames (if found).
    """
    print("Loading train image and label filenames...")
    
    files = os.listdir(training_path)
    image_files = sorted([f for f in files
                         if f.endswith(image_format)
                         and "label" not in f.lower()
                         and "mask" not in f.lower()
                         and "integer" not in f.lower()
                         and re.search(r'\d+', f)])
    
    label_files = sorted([f for f in files
                         if f.endswith(label_format)
                         and "heat" in f.lower()
                         and re.search(r'\d+', f)])
    
    integer_label_files = sorted([f for f in files
                                 if (f.endswith(".npy") or f.endswith(".tif"))
                                 and "integer" in f.lower()
                                 and re.search(r'\d+', f)])
    
    assert len(image_files) == len(label_files), "Number of image and label files do not match."
    assert len(image_files) == len(integer_label_files), "Number of image and integer label files do not match."
    return image_files, label_files, integer_label_files