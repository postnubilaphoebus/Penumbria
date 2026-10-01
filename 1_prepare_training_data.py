import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import skimage
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt, gaussian_filter, find_objects
import tifffile
from tqdm import tqdm
from fractions import Fraction

from volume_io import coerce_resizing_factors, imagej_label_dtype, scaled_shape


SUPPORTED_VOLUME_SUFFIXES = {".tif", ".tiff", ".npy"}


def _pairing_key(filename, role_filter):
    """Remove a role token and normalise separators to identify one sample."""
    stem = Path(filename).stem
    match = re.search(re.escape(role_filter), stem, flags=re.IGNORECASE)
    if match is None:
        return None
    without_role = stem[:match.start()] + stem[match.end():]
    return re.sub(r"[^a-z0-9]+", "_", without_role.casefold()).strip("_")


def _index_by_pairing_key(paths, role_filter, role_name):
    indexed = {}
    for path in paths:
        key = _pairing_key(path.name, role_filter)
        if not key:
            raise ValueError(
                f"Cannot derive a sample key from {role_name} file {path.name!r} "
                f"using filter {role_filter!r}"
            )
        if key in indexed:
            raise ValueError(
                f"Multiple {role_name} files resolve to sample key {key!r}: "
                f"{indexed[key].name!r} and {path.name!r}"
            )
        indexed[key] = path
    return indexed


def discover_training_pairs(base_path, img_filter="img", label_filter="label"):
    """Return deterministic, explicitly key-matched image/label file pairs."""
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise NotADirectoryError(f"Training-data folder does not exist: {base_path}")
    if not img_filter or not label_filter:
        raise ValueError("Image and label filters must both be non-empty")

    volume_paths = sorted(
        (path for path in base_path.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_VOLUME_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    image_paths = [path for path in volume_paths if img_filter.casefold() in path.stem.casefold()]
    label_paths = [path for path in volume_paths if label_filter.casefold() in path.stem.casefold()]
    ambiguous = sorted(set(image_paths) & set(label_paths))
    if ambiguous:
        raise ValueError(
            "Files match both image and label filters: " + ", ".join(path.name for path in ambiguous)
        )
    unclassified = [path for path in volume_paths if path not in image_paths and path not in label_paths]
    if unclassified:
        raise ValueError(
            "Volume files match neither filter; move them out or choose explicit filters: "
            + ", ".join(path.name for path in unclassified)
        )
    if not image_paths or not label_paths:
        raise ValueError(
            f"Found {len(image_paths)} image files and {len(label_paths)} label files in {base_path}"
        )

    images_by_key = _index_by_pairing_key(image_paths, img_filter, "image")
    labels_by_key = _index_by_pairing_key(label_paths, label_filter, "label")
    missing_labels = sorted(set(images_by_key) - set(labels_by_key))
    missing_images = sorted(set(labels_by_key) - set(images_by_key))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append("images without labels: " + ", ".join(missing_labels))
        if missing_images:
            details.append("labels without images: " + ", ".join(missing_images))
        raise ValueError("Image/label pairing failed; " + "; ".join(details))

    return [(key, images_by_key[key], labels_by_key[key]) for key in sorted(images_by_key)]


def prepare_empty_output_directory(path):
    """Create and lock *path*, refusing any directory that already has content."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        first_entry = next(path.iterdir(), None)
        if first_entry is not None:
            raise FileExistsError(
                f"Refusing to write into non-empty output directory {path}. "
                "Choose a new --output_path or empty the directory explicitly."
            )
    else:
        path.mkdir(parents=True)
    lock_path = path / ".penumbria_prepare.lock"
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Another preparation process already locked {path}") from error
    return path


def load_volume(path, dtype):
    if path.suffix.casefold() == ".npy":
        return np.load(path).astype(dtype)
    return skimage.io.imread(path).astype(dtype)


def write_imagej_volume(path, values, dtype):
    """Write one complete ImageJ stack before publishing its final filename."""
    path = Path(path)
    dtype = np.dtype(dtype)
    partial_path = path.with_name(f".{path.name}.partial.tif")
    tifffile.imwrite(
        partial_path,
        np.asarray(values, dtype=dtype),
        bigtiff=False,
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    replace_with_retry(partial_path, path)


def replace_with_retry(source, destination, attempts=8):
    """Publish a completed file despite transient Windows scanner locks."""
    delay = 0.05
    for attempt in range(attempts):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def write_json_atomic(path, payload):
    path = Path(path)
    partial_path = path.with_name(f".{path.name}.partial")
    partial_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    replace_with_retry(partial_path, path)


def comma_fraction_list(s):
    return list(coerce_resizing_factors(s.split(",")))

def format_factor(x):
    return f"{float(x):.2f}".replace('.', 'p')

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
        type=comma_fraction_list,
        help='Positive decimal or fractional resizing factors (eg --resizing_factors=1/3,1,1)',
        default=[1.0, 1.0, 1.0] # Set the default to the list [1.0, 1.0, 1.0]
    )
    parser.add_argument('--base_path', type=str, help='Base path to images and masks')
    parser.add_argument('--output_path', type=str, help='Output path for heatmaps', default="prepped_data")
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--minimum_foreground_label', type=int, help='Minimum Label considered Foreground') 
    # will be 2 for arabidopsis dataset, normally is 1
    parser.add_argument('--img_filter', type=str, help='Image filter', default="img")
    parser.add_argument('--label_filter', type=str, help='Label filter', default="label")
    args = parser.parse_args()

    assert args.dataset_name is not None
    assert args.base_path is not None

    options = vars(args)
    print(options)
    resizing_factors = coerce_resizing_factors(args.resizing_factors)
    if len(resizing_factors) != 3 or any(factor <= 0 for factor in resizing_factors):
        raise ValueError("--resizing_factors must contain three positive values")

    folder_name = args.output_path + "_" + args.dataset_name + "_resizing_" + "_".join(format_factor(x) for x in resizing_factors)
    pairs = discover_training_pairs(args.base_path, args.img_filter, args.label_filter)
    saving_path = prepare_empty_output_directory(folder_name)

    print("matched image/label pairs:")
    for index, (key, image_path, label_path) in enumerate(pairs):
        print(f"  {index}: {image_path.name} <-> {label_path.name} (key={key})")
    print("inference resolution resizing factors:", resizing_factors)
    background_value = -5.0

    manifest = {
        "version": 2,
        "status": "in_progress",
        "base_path": str(Path(args.base_path).resolve()),
        "dataset_name": args.dataset_name,
        "img_filter": args.img_filter,
        "label_filter": args.label_filter,
        "resizing_factors": [float(factor) for factor in resizing_factors],
        "resizing_factor_fractions": [str(factor) for factor in resizing_factors],
        "minimum_foreground_label": args.minimum_foreground_label,
        "completed_samples": [],
        "pairs": [
            {
                "index": index,
                "key": key,
                "image": image_path.name,
                "label": label_path.name,
                "image_size": image_path.stat().st_size,
                "label_size": label_path.stat().st_size,
                "image_mtime_ns": image_path.stat().st_mtime_ns,
                "label_mtime_ns": label_path.stat().st_mtime_ns,
            }
            for index, (key, image_path, label_path) in enumerate(pairs)
        ],
    }
    manifest_path = saving_path / "preparation_manifest.json"
    write_json_atomic(manifest_path, manifest)

    for hhh, (sample_key, source_image_path, source_label_path) in enumerate(tqdm(pairs)):
        label_img = load_volume(source_label_path, np.int32)
        img = load_volume(source_image_path, np.float32)
        if label_img.ndim != 3 or img.ndim != 3:
            raise ValueError(
                f"Only 3-D data are supported for {sample_key!r}; "
                f"image shape={img.shape}, label shape={label_img.shape}"
            )
        if label_img.shape != img.shape:
            raise ValueError(
                f"Image/label shape mismatch for {sample_key!r}: "
                f"{source_image_path.name} has {img.shape}, {source_label_path.name} has {label_img.shape}"
            )
        source_shape = img.shape
        if any(f != 1.0 for f in resizing_factors):
            output_shape = scaled_shape(label_img.shape, resizing_factors)
            label_img = resize(
                label_img, output_shape, order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.int32, copy=False)
            img = resize(img, output_shape, order=3, preserve_range=True).astype(np.float32, copy=False)
        minimum_label = int(label_img.min())
        maximum_label = int(label_img.max())
        if minimum_label < 0:
            raise ValueError(
                f"Negative label ID {minimum_label} in {source_label_path.name}; "
                "ImageJ label TIFFs require non-negative IDs"
            )
        integer_output_dtype = imagej_label_dtype(maximum_label)

        label_heat = np.ones_like(label_img).astype(np.float32) * (background_value)
        # Instance IDs can be sparse (e.g. ~600 real instances spread across
        # IDs up to 60k+), so don't probe every integer in
        # [minimum_foreground_label, maximum_label] with its own full-volume
        # np.argwhere scan. np.unique() finds which IDs actually exist and
        # find_objects() computes every instance's bounding-box slice in a
        # single pass over the whole array -- both O(volume) once, total,
        # instead of O(volume) per candidate ID.
        present_ids = np.unique(label_img)
        present_ids = present_ids[present_ids >= args.minimum_foreground_label]
        object_slices = find_objects(label_img, max_label=maximum_label)
        for i in tqdm(present_ids):
            slc = object_slices[i - 1]  # find_objects() is 1-indexed
            if slc is None:
                continue
            # Same 1-voxel zero margin as the original tight-bbox+3
            # construction, on every side, even at the real volume edge --
            # NOT clamped away. twice_smooth_and_threshold()'s gaussian_filter
            # uses reflect-mode boundary handling, which is sensitive to the
            # box's exact size/edge position, so dropping that margin row at
            # a real edge silently changes the smoothed mask there. Instead,
            # keep the full padded box and only copy the in-bounds portion
            # from label_img, leaving any margin beyond the real volume at
            # its zero-initialized value (exactly what the original's fresh
            # np.zeros(...) bounding_box already did).
            starts = [s.start - 1 for s in slc]
            stops = [s.stop + 1 for s in slc]
            box_shape = tuple(b - a for a, b in zip(starts, stops))
            src_slc = tuple(
                slice(max(a, 0), min(b, dim))
                for a, b, dim in zip(starts, stops, label_img.shape)
            )
            dst_slc = tuple(
                slice(max(a, 0) - a, (max(a, 0) - a) + (min(b, dim) - max(a, 0)))
                for a, b, dim in zip(starts, stops, label_img.shape)
            )
            bounding_box = np.zeros(box_shape, dtype=np.float32)
            bounding_box[dst_slc] = (label_img[src_slc] == i)
            if not np.any(bounding_box):
                continue
            heat_values, points_to_return = transform_shape_to_edt(bounding_box)
            offset = np.array(starts)
            points_to_return = points_to_return + offset
            if heat_values is not None and points_to_return.size > 3 and heat_values.size > 0:
                label_heat[points_to_return[:, 0], points_to_return[:, 1], points_to_return[:, 2]] = heat_values

        heat_path = saving_path / f"{hhh}heat_mask.tif"
        integer_path = saving_path / f"{hhh}integer.tif"
        prepared_image_path = saving_path / f"{hhh}img.tif"
        write_imagej_volume(heat_path, label_heat, np.float32)
        write_imagej_volume(integer_path, label_img, integer_output_dtype)
        write_imagej_volume(prepared_image_path, img, np.float32)
        manifest["pairs"][hhh]["source_shape"] = list(source_shape)
        manifest["pairs"][hhh]["prepared_shape"] = list(img.shape)
        manifest["pairs"][hhh]["effective_resizing_fractions"] = [
            str(Fraction(prepared, source))
            for source, prepared in zip(source_shape, img.shape)
        ]
        manifest["completed_samples"].append(hhh)
        write_json_atomic(manifest_path, manifest)

    # Use the clean, consistent resizing_factors variable
    with open(saving_path / "resizing_factors.txt", "w") as f:
        f.write("[" + ", ".join(str(factor) for factor in resizing_factors) + "]")
    manifest["status"] = "complete"
    write_json_atomic(manifest_path, manifest)
    (saving_path / ".penumbria_prepare.lock").unlink()
