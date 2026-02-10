import argparse
import os

import numpy as np
import skimage
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt, gaussian_filter
import tifffile
from tqdm import tqdm

def comma_float_list(s):
    return [float(x) for x in s.split(",")]

def format_factor(x):
    return f"{x:.2f}".replace('.', 'p')

def transform_shape_to_edt(img, low_value = -2.0, high_value = 20.0):
    original_image = img.copy()
    # 1. remove label impurities:
    img = twice_smooth_and_threshold(img)
    points_to_return = np.argwhere(img)
    # 2. Check if label disappears after smoothing
    if points_to_return.size > 0:
        x_range, y_range, z_range = points_to_return[:, 0].max() - points_to_return[:, 0].min(), \
                                    points_to_return[:, 1].max() - points_to_return[:, 1].min(), \
                                    points_to_return[:, 2].max() - points_to_return[:, 2].min()
    else:
        img = original_image
        points_to_return = np.argwhere(img)
        x_range, y_range, z_range = points_to_return[:, 0].max() - points_to_return[:, 0].min(), \
                                    points_to_return[:, 1].max() - points_to_return[:, 1].min(), \
                                    points_to_return[:, 2].max() - points_to_return[:, 2].min()
    # 3. Check if label is NOT a pancake cell (too small in one dimension)
    if x_range > 1 and y_range > 1 and z_range > 1:
        dist_trans = distance_transform_edt(img)
        ordered_bool_image_2_new = dist_trans
    else:
        # label is a pancake cell
        # edt transform happens per slice
        if x_range > 0:
            ranges = np.array([x_range, y_range, z_range])
            minimum_axis = np.argmin(ranges)
            axes_without_minimum = np.array([axis for idx, axis in enumerate(ranges) if idx != minimum_axis])
            ordered_bool_image_2_new = np.zeros_like(img).astype(np.float32)
            if not np.all(ranges[minimum_axis] < axes_without_minimum):
                minimum_axis = 0
            if minimum_axis == 0:
                x_shape, y_shape, z_shape = img.shape
                for xxx in range(x_shape):
                    current_slice = img[xxx, :, :]
                    if current_slice.sum() > 0:
                        heat2d_image = distance_transform_edt(current_slice)
                        ordered_bool_image_2_new[xxx, :, :] = heat2d_image
            elif minimum_axis == 1:
                x_shape, y_shape, z_shape = img.shape
                for yyy in range(y_shape):
                    current_slice = img[:, yyy, :]
                    if current_slice.sum() > 0:
                        heat2d_image = distance_transform_edt(current_slice)
                        ordered_bool_image_2_new[:, yyy, :] = heat2d_image
            else:
                x_shape, y_shape, z_shape = img.shape
                for zzz in range(z_shape):
                    current_slice = img[:, :, zzz]
                    if current_slice.sum() > 0:
                        heat2d_image = distance_transform_edt(current_slice)
                        ordered_bool_image_2_new[:, :, zzz] = heat2d_image
        else:
            ordered_bool_image_2_new = np.zeros_like(img).astype(np.float32)
            x_vals = np.unique(points_to_return[:, 0])
            y_vals = np.unique(points_to_return[:, 1])
            z_vals = np.unique(points_to_return[:, 2])
            unique_val_sizes = np.array([x_vals.size, y_vals.size, z_vals.size])
            minimum_axis = np.argmin(unique_val_sizes)
            axes_without_minimum = np.array([axis for idx, axis in enumerate(unique_val_sizes) if idx != minimum_axis])
            ordered_bool_image_2_new = np.zeros_like(img).astype(np.float32)
            if not np.all(unique_val_sizes[minimum_axis] < axes_without_minimum):
                minimum_axis = 0
            if minimum_axis == 0:
                x_slice = img[x_vals[0], :, :]
                heat2d_image = distance_transform_edt(x_slice)
                ordered_bool_image_2_new[x_vals[0], :, :] = heat2d_image
            elif minimum_axis == 1:
                y_slice = img[:, y_vals[0], :]
                heat2d_image = distance_transform_edt(y_slice)
                ordered_bool_image_2_new[:, y_vals[0], :] = heat2d_image
            else:
                z_slice = img[:, :, z_vals[0]]
                heat2d_image = distance_transform_edt(z_slice)
                ordered_bool_image_2_new[:, :, z_vals[0]] = heat2d_image

    heat_values = ordered_bool_image_2_new[img > 0]
    if heat_values.min() == heat_values.max():
        return np.ones(points_to_return.shape[0]) * 20.0, points_to_return
    else:
        heat_values = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min())
        heat_values = low_value + (high_value - low_value) * (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min())
        return heat_values, points_to_return

def twice_smooth_and_threshold(img):
    gfilt = gaussian_filter(img, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    gfilt = gaussian_filter(gfilt, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    return gfilt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='label_to_heatmap',
        description='Heatmap generation and upsampling for labels'
    )
    parser.add_argument(
        '--resizing_factors', 
        type=comma_float_list, 
        help='Resizing factors to use for up- or downsampling (eg --resizing_factors=7,1,1)', 
        default=[1.0, 1.0, 1.0] # Set the default to the list [1.0, 1.0, 1.0]
    ) 
    parser.add_argument('--base_path', type=str, help='Base path to images and masks')
    parser.add_argument('--output_path', type=str, help='Output path for heatmaps', default="prepped_data")
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--img_filter', type=str, help='Image filter', default="img")
    args = parser.parse_args()

    assert args.dataset_name is not None
    assert args.base_path is not None

    options = vars(args)
    print(options)
    resizing_factors = args.resizing_factors 
    
    folder_name = args.output_path + "_" + args.dataset_name + "_resizing_" + "_".join(format_factor(x) for x in resizing_factors)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    path = args.base_path
    img_filter = args.img_filter
    label_names = os.listdir(path)

    print("only keeping files with .tif or .npy extension")
    label_names = [name for name in label_names if name.endswith(".tif") or name.endswith(".npy")]
    img_names = sorted([name for name in label_names if img_filter in name])
    label_names = sorted([name for name in label_names if img_filter not in name])

    print("label names:", label_names)
    print("inference resolution resizing factors:", resizing_factors)
    base_path = path
    background_value = -5.0
    minimum_foreground_label = 1 # will be 2 for arabidopsis dataset

    print("minimum foreground label:", minimum_foreground_label)
    print("some datasets like arabidopsis have 1 as the background, so check the output!")

    kkk = 0
    for label_name in tqdm(label_names):
        image_path = os.path.join(base_path, label_name)
        actual_image_path = os.path.join(base_path, img_names[kkk])
        try:
            label_img = skimage.io.imread(image_path).astype(np.int32)
        except:
            label_img = np.load(image_path).astype(np.int32)
        img = skimage.io.imread(actual_image_path).astype(np.float32)
        if any(f != 1.0 for f in resizing_factors):
            label_img = resize(label_img, (label_img.shape[0] * resizing_factors[0], 
                                           label_img.shape[1] * resizing_factors[1], 
                                           label_img.shape[2] * resizing_factors[2]), 
                                           order = 0)
            img = resize(img, (img.shape[0] * resizing_factors[0], 
                               img.shape[1] * resizing_factors[1], 
                               img.shape[2] * resizing_factors[2]), 
                               order = 3)

        label_heat = np.ones_like(label_img).astype(np.float32) * (background_value)
        for i in tqdm(range(minimum_foreground_label, label_img.max() + 1)):
            ppoints = np.argwhere(label_img == i)
            if ppoints.size < 3:
                continue
            min_x = ppoints[:, 0].min()
            max_x = ppoints[:, 0].max()
            min_y = ppoints[:, 1].min()
            max_y = ppoints[:, 1].max()
            min_z = ppoints[:, 2].min()
            max_z = ppoints[:, 2].max()
            bounding_box = np.zeros((max_x - min_x + 3, max_y - min_y + 3, max_z - min_z + 3))
            ppoints_shifted = ppoints - np.array([min_x - 1, min_y - 1, min_z - 1])
            bounding_box[ppoints_shifted[:, 0], ppoints_shifted[:, 1], ppoints_shifted[:, 2]] = 1
            heat_values, points_to_return = transform_shape_to_edt(bounding_box)
            points_to_return += np.array([min_x - 1, min_y - 1, min_z - 1])
            if heat_values is not None and points_to_return.size > 3 and heat_values.size > 0:
                label_heat[points_to_return[:, 0], points_to_return[:, 1], points_to_return[:, 2]] = heat_values
                
        name_prefix, file_extension = os.path.splitext(label_name)
        new_name = str(kkk) + "heat_mask.tif"
        saving_path = os.path.join(base_path, folder_name)
        label_path = os.path.join(saving_path, new_name)

        tifffile.imwrite(label_path, label_heat.astype(np.float32))
        tifffile.imwrite(label_path.replace("heat_mask", "integer"), label_img.astype(np.float32))
        tifffile.imwrite(label_path.replace("heat_mask", "img"), img.astype(np.float32))
        kkk += 1

    # Use the clean, consistent resizing_factors variable
    with open(os.path.join(folder_name, "resizing_factors.txt"), "w") as f:
        f.write(str(resizing_factors))