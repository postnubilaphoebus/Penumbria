import torch
import numpy as np
from utils import (
    load_inference_images,
    load_training_images_and_labels,
    load_training_source_shapes,
    preprocess_0_1,
)
import os
from train import train_model
from inference import sliding_window_inference
from skimage.transform import resize
import warnings
import tifffile
from typing import Dict
import yaml
import sys
import copy
import random
import argparse
from pathlib import Path
from model_config import add_model_arguments, build_model, with_model_defaults
from volume_io import (
    export_zarr_to_tiff,
    resize_volume_to_ome_zarr,
    scaled_shape,
    select_inference_resizing_factors,
)


def load_config(config_path: str) -> Dict:
    """loads the yaml config file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def override_config(config, args):
    """Apply CLI argument overrides to nested config dictionary"""
    merged_config = copy.deepcopy(config)

    for key, value in vars(args).items():
        if value is None or key in ('config', 'run_id'):
            continue

        # Find which section contains this key
        for section_name, section_content in merged_config.items():
            if isinstance(section_content, dict) and key in section_content:
                merged_config[section_name][key] = value
                break

    return merged_config


def main(seed):

    ###################################################################################################################
    ######################################### configuration and initialization ########################################
    ###################################################################################################################

    # You may load a specific yaml file using the -c argument
    # For example: python 1_train_and_infer.py -c="./dataset_configs/zebrafish_confocal.yaml"
    # However, you may override those using command line arguments

    # ───────────────────────────────────────────────────────────────
    # set random seeds for reproducibility

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ───────────────────────────────────────────────────────────────
    # program name

    parser = argparse.ArgumentParser(
        prog='Penumbria',
        description='Heatmap neural network for cell segmentation - training + inference'
    )

    # ───────────────────────────────────────────────────────────────
    # global config file
    parser.add_argument(
        '-c', '--config',
        type=str,
        default = "default_config.yaml",
        help='Path to YAML configuration file'
    )

    # ───────────────────────────────────────────────────────────────
    # run identifier, so repeated runs in the same folder don't overwrite each other.
    # Also pass this same value to 2_watershed_tune.py so it can find these outputs.
    parser.add_argument('--run_id', type=str, default="",
        help='Identifier appended to output filenames for this run (e.g. "run1")')

    # ───────────────────────────────────────────────────────────────
    # label_transform section

    parser.add_argument('--high_value', type=float,
        help='High value for cell heatmap (positive peak)')
    parser.add_argument('--low_value', type=float,
        help='Low value for cell heatmap (negative peak)')
    parser.add_argument('--background_maximum', type=float,
        help='Maximum background heatmap intensity')
    parser.add_argument('--foreground_minimum', type=float,
        help='Minimum foreground (cell) heatmap intensity')
    parser.add_argument('--ignore_index', type=int,
        help='Index to ignore during loss function computation')

    # ───────────────────────────────────────────────────────────────
    # model section

    parser.add_argument('--optimizer', type=str,
        help='Optimizer (e.g., sgd, adam)')
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate for optimizer')
    parser.add_argument('--load_pretrained', type=bool,
        help='Whether to load pretrained model weights')
    parser.add_argument('--momentum', type=float,
        help='Momentum for SGD optimizer, if applicable')
    parser.add_argument('--model_weights_path', type=str,
        help='Path to model weights file')
    add_model_arguments(parser)

    # ───────────────────────────────────────────────────────────────
    # training section
    parser.add_argument('--training_iterations', type=int,
        help='Number of training iterations')
    parser.add_argument('--evaluation_interval', type=int,
        help='Number of iterations between evaluation')
    parser.add_argument('--data_dimensionality', type=int,
        help='Data dimensionality: 2D or 3D')
    parser.add_argument('--mixed_precision', type=bool,
        help='Use mixed precision for training (true/false)')
    parser.add_argument('--val_indices', type=int, nargs='+',
        help='Indices of validation images')
    parser.add_argument('--dynamic_cropping', type=bool,
        help='Use random patch sampling instead of fixed cropping')
    parser.add_argument('--training_image_shape', type=int, nargs='+',
        help='Training input patch size (e.g., 64 64 64)')
    parser.add_argument('--verbosity_flag', type=bool,
        help='Enable verbose output (true/false)')
    parser.add_argument('--data_augmentation_types', type=str, nargs='+',
        help='List of augmentation types to apply during training')
    parser.add_argument('--mini_batch_size', type=int,
        help='Mini-batch size')
    parser.add_argument('--early_stopping_patience', type=int,
        help='Epochs with no improvement before early stopping triggers')
    parser.add_argument('--training_folder', type=str,
        help='Path to training input folder (images and labels)')
    parser.add_argument('--inference_folder', type=str,
        help='Where images to segment are located')
    parser.add_argument('--inference_resolution_upsampling', type=float, nargs='+',
        help='Override inherited training resize factors during inference (e.g., 1 0.5 0.5)')

    # ───────────────────────────────────────────────────────────────
    # inference section

    parser.add_argument('--test_time_augmentation', type=bool,
        help='Use augmentation during inference (tile averaging)')
    parser.add_argument('--keep_size', type=int, nargs='+',
        help='Patch size to keep (e.g., 12 60 60)')
    parser.add_argument('--step_size', type=int, nargs='+',
        help='Stride/step size for inference tiling')
    parser.add_argument('--inference_indices', type=int, nargs='+',
        help='Indices of images to segment')

    # ───────────────────────────────────────────────────────────────
    # postprocessing section (only 'parameter_tuning' is used here, to decide
    # whether validation heatmaps need to be produced for 2_watershed_tune.py)

    parser.add_argument('--parameter_tuning', type=int,
        help='whether parameter tuning will be performed downstream on validation data')

    args = parser.parse_args()
    args_dict = vars(args)
    config_path = args_dict.get("config")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    # Older configs may not list the architecture options; fill them in so CLI overrides apply
    config['model'] = with_model_defaults(config['model'])

    # Apply CLI overrides
    merged_config = override_config(config, args)

    # Extract configuration sections
    label_cfg = merged_config['label_transform']
    model_cfg = merged_config['model']
    train_cfg = merged_config['training']
    inference_cfg = merged_config['inference']
    post_cfg = merged_config['postprocessing']

    # run identifier: appended to output filenames so repeated runs in the same
    # output folder don't overwrite each other. Pass the same value to 2_watershed_tune.py.
    run_id = args.run_id
    tag = f"_{run_id}" if run_id else ""

    # Label transform
    high_value = label_cfg['high_value']
    low_value = label_cfg['low_value']
    background_maximum = label_cfg['background_maximum']
    foreground_minimum = label_cfg['foreground_minimum']
    ignore_index = label_cfg['ignore_index']

    # Model
    chosen_optimizer = model_cfg['optimizer']
    learning_rate = model_cfg['learning_rate']
    load_pretrained = model_cfg['load_pretrained']
    momentum = model_cfg['momentum']
    model_weights_path = model_cfg['model_weights_path']
    use_cell_hint = model_cfg['use_cell_hint']

    # Training
    training_iterations = train_cfg['training_iterations']
    evaluation_interval = train_cfg['evaluation_interval']
    data_dimensionality = train_cfg['data_dimensionality']
    mixed_precision = train_cfg['mixed_precision']
    dynamic_cropping = train_cfg['dynamic_cropping']
    training_image_shape = train_cfg['training_image_shape']
    verbosity_flag = train_cfg['verbosity_flag']
    data_augmentation_types = train_cfg['data_augmentation_types']
    mini_batch_size = train_cfg['mini_batch_size']
    early_stopping_patience = train_cfg['early_stopping_patience']
    training_path = train_cfg['training_folder']
    inference_path = train_cfg['inference_folder']
    inference_resolution_upsampling = train_cfg['inference_resolution_upsampling']
    in_channels = train_cfg['in_channels']
    val_indices = train_cfg['val_indices']

    # Inference
    test_time_augmentation = inference_cfg['test_time_augmentation']
    keep_size = inference_cfg['keep_size']
    step_size = inference_cfg['step_size']
    predicted_label_path = inference_path
    inference_indices = inference_cfg['inference_indices']

    # Postprocessing (only used to decide whether to emit validation heatmaps)
    parameter_tuning = post_cfg['parameter_tuning']

    assert foreground_minimum >= background_maximum, "foreground value cannot be lower than any background"
    assert low_value < background_maximum, "low value must be less than background maximum"
    assert low_value < foreground_minimum, "low value must be less than foreground minimum"
    assert high_value > low_value, "high value must be greater than low value"
    assert high_value > foreground_minimum, "high value must be greater than foreground minimum"
    assert mini_batch_size > 0, "mini batch size must be at least 1"
    assert early_stopping_patience > 0, "early stopping patience must be at least 1"
    assert learning_rate > 0, "learning rate must be at least 0"
    assert training_path is not None, "training path must be specified"
    assert os.path.exists(training_path), f"training path does not exist {training_path}"
    if inference_path is not None:
        assert os.path.exists(inference_path), f"inference path does not exist {inference_path}"
    else:
        print("inference path is None, using sampled training image as inference image")

    train_shape_arr = np.array(training_image_shape)
    all_same = np.all(train_shape_arr == train_shape_arr[0])
    if not all_same:
        raise ValueError(f"training image shape must be cube, currently: {train_shape_arr}.\
                         consider resampling your whole image in case of anisotropy.")

    five_divided = train_shape_arr[0] // (2 ** 5)
    if five_divided * (2 ** 5) != train_shape_arr[0]:
        raise ValueError("training image shape must be divisible by 2^5 due to model choice (Uvixlstm)")

    if train_shape_arr[0] > 192:
        warnings.warn(f"training image shape is large ({train_shape_arr[0]} cubed), this may give OOM errors.")

    ###################################################################################################################
    ##################################### data preprocessing and model initialization #################################
    ###################################################################################################################

    images, labels, integer_labels, resizing_factors = load_training_images_and_labels(training_path,
                                                                     image_format=".tif",
                                                                     label_format=".tif")
    training_source_shapes = load_training_source_shapes(
        training_path, [image.shape for image in images], resizing_factors
    )

    resizing_necessary = [1.0 != f for f in resizing_factors]
    resizing_necessary = np.array(resizing_necessary)

    if np.any(resizing_necessary):
        print("resizing necessary")
        print("resizing factors:", resizing_factors)
    else:
        print("resizing not necessary")

    print("training images use cached exact whole-volume normalisation statistics")

    external_inference = inference_path is not None
    if not external_inference:
        inference_images = [images[idx] for idx in inference_indices]
        inference_filenames = [str(idx) for idx in inference_indices]
        inference_target_shapes = [training_source_shapes[idx] for idx in inference_indices]
        current_path = os.getcwd()
        project_path_inference = os.path.join(current_path, "inference_data")
        predicted_label_path = project_path_inference

    else:
        inference_images, inference_filenames = load_inference_images(
            inference_path, fileformat=".tif", normalize=True
        )
        inference_target_shapes = [image.shape for image in inference_images]
        inference_indices = [-10000000, -20000000]


    print("inference filenames", inference_filenames)
    if len(inference_images) == 0:
        raise FileNotFoundError(f"No .tif or .tiff inference images found in {inference_path}")
    assert all(getattr(image, "normalization", None) is not None for image in images), \
           "training images must carry whole-volume normalisation statistics"
    assert all(getattr(image, "normalization", None) is not None for image in inference_images), \
           "inference images must carry whole-volume normalisation statistics"

    mask_file_matrix = []
    mask_filename_matrix = []

    if inference_resolution_upsampling is not None:
        inference_resize_factors = select_inference_resizing_factors(
            resizing_factors, inference_resolution_upsampling
        )
        resize_source = "explicit inference_resolution_upsampling override"
    elif external_inference:
        inference_resize_factors = select_inference_resizing_factors(resizing_factors, None)
        resize_source = "inherited training resizing_factors"
    else:
        inference_resize_factors = select_inference_resizing_factors(
            (1,) * len(resizing_factors), None
        )
        resize_source = "identity factors for already-prepared cross-validation images"

    print(f"inference resize factors ({resize_source}): "
          f"{[str(factor) for factor in inference_resize_factors]}")
    if any(factor != 1 for factor in inference_resize_factors):
        print("resizing inference images into model resolution...")
        resized_images = []
        for name, image in zip(inference_filenames, inference_images):
            output_shape = scaled_shape(image.shape, inference_resize_factors)
            output_path = os.path.join(
                inference_path if external_inference else predicted_label_path,
                ".penumbria_zarr",
                f"{name}-inference-resized.ome.zarr",
            )
            resized_images.append(resize_volume_to_ome_zarr(
                image, output_path, output_shape, order=3, is_label=False
            ))
        inference_images = resized_images

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels_intensity = labels

    apply_padding = dynamic_cropping

    print("pad images for training (if dynamic cropping enabled)...")

    # Validate dimensions
    if len(training_image_shape) not in [2, 3]:
        raise ValueError(f"Incorrect training image shape provided: {training_image_shape}")

    # Calculate padding
    maximum_training_image_dim = max(training_image_shape)
    pad_length = maximum_training_image_dim if apply_padding else int(maximum_training_image_dim / 2)

    # ChunkedVolume performs reflected/constant padding only on requested
    # patches.  These references are therefore both padded views and unpadded
    # sources without allocating complete copies.
    images_unpadded = images.copy()
    integer_labels_unpadded = integer_labels.copy()

    if apply_padding:
        print("using lazy reflected patch padding (integer-label padding = -100)")
    else:
        print("dynamic cropping disabled; complete images may still need to fit in RAM/GPU")

    print("val indices", val_indices)
    val_images = [images[i] for i in val_indices]
    input_images = [images[i] for i in range(len(images)) if i not in val_indices and i not in inference_indices]
    train_labels_intensity = [labels_intensity[i] for i in range(len(labels_intensity)) \
                              if i not in val_indices and i not in inference_indices]
    val_labels_intensity = [labels_intensity[i] for i in val_indices]
    train_labels_integer = [integer_labels[i] for i in range(len(integer_labels)) \
                            if i not in val_indices and i not in inference_indices]
    val_labels_integer = [integer_labels[i] for i in val_indices]
    val_images_unpadded = [images_unpadded[i] for i in val_indices]
    val_labels_integer_unpadded = [integer_labels_unpadded[i] for i in val_indices]


    if not load_pretrained:

        ################################################################################################################
        ##################################### model training ###########################################################
        ################################################################################################################

        print("beginning training...")
        print("training_image_shape", training_image_shape)
        model = build_model(model_cfg, training_image_shape[0]).to(device)
        print("model options:", {key: model_cfg[key] for key in
                                 ("use_sgg_layer", "use_cell_hint", "zernike_enabled", "zernike_moments")})
        model.train()
        loss_fn = torch.nn.MSELoss(reduction="none")

        if chosen_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                                            nesterov=False, weight_decay=1e-4)
        elif chosen_optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.9))
        elif chosen_optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.9), weight_decay=1e-5)
        else:
            raise ValueError("unknown optimizer: {}".format(chosen_optimizer))

        model = train_model(model,
                            optimizer,
                            loss_fn,
                            input_images,
                            val_images,
                            train_labels_intensity,
                            val_labels_intensity,
                            train_labels_integer,
                            val_labels_integer,
                            early_stopping_patience,
                            device,
                            pad_length,
                            mixed_precision = mixed_precision,
                            ignore_index = ignore_index,
                            dynamic_cropping = dynamic_cropping,
                            training_image_shape = training_image_shape,
                            keep_size = keep_size,
                            mini_batch_size = mini_batch_size,
                            verbose = verbosity_flag,
                            training_iterations = training_iterations,
                            data_augmentation_types = data_augmentation_types,
                            evaluation_interval = evaluation_interval,
                            use_cell_hint = use_cell_hint)

        weights_path = Path(model_weights_path)
        model_weights_path = str(weights_path.with_name(f"{seed}_{weights_path.name}"))
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_weights_path)

    else:
        print("loading model weights...")
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError("model weights not found at {}".format(model_weights_path))
        # The model options in the config must match the ones the weights were trained with.
        model = build_model(model_cfg, training_image_shape[0]).to(device)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))

    ####################################################################################################################
    ##################################### model inference ##############################################################
    ####################################################################################################################

    # Never normalise individual crops: bright foreground-only crops would
    # otherwise destroy the background statistics learned during training.
    patch_based_norm = False

    print("beginning inference...")
    model.eval()
    model_prediction, inference_filenames, padding_list_inferece = sliding_window_inference(model,
                                                                     inference_images,
                                                                     False,
                                                                     mask_file_matrix,
                                                                     mask_filename_matrix,
                                                                     device,
                                                                     -5.0,
                                                                     high_value,
                                                                     predicted_label_path,
                                                                     inference_filenames,
                                                                     mixed_precision,
                                                                     patch_based_norm,
                                                                     tta = test_time_augmentation,
                                                                     image_dim = training_image_shape,
                                                                     keep_size = keep_size,
                                                                     step_size = step_size)

    # Always resize to the exact source shape and export - 2_watershed_tune.py reads
    # these "{name}_inference_output_resized" files back from disk.
    print("restoring predictions to exact source shapes and saving heatmaps...")
    resized_predictions = []
    for name, prediction, output_shape in zip(
        inference_filenames, model_prediction, inference_target_shapes
    ):
        output_path = os.path.join(
            predicted_label_path, "preds", f"{name}_inference_output_resized{tag}.ome.zarr"
        )
        resized_prediction = resize_volume_to_ome_zarr(
            prediction, output_path, output_shape, order=3, is_label=False
        )
        resized_predictions.append(resized_prediction)
        export_zarr_to_tiff(
            resized_prediction,
            os.path.join(predicted_label_path, "preds", f"{name}_inference_output_resized{tag}.tif"),
            output_dtype=np.float32,
        )
    model_prediction = resized_predictions

    ####################################################################################################################
    ##################################### validation heatmaps (for 2_watershed_tune.py) ################################
    ####################################################################################################################

    if parameter_tuning:

        print("producing validation heatmaps for watershed tuning...")
        val_heatmaps, __, padding_list_val = sliding_window_inference(model,
                                                                      val_images_unpadded,
                                                                      False,
                                                                      None,
                                                                      None,
                                                                      device,
                                                                      -5.0,
                                                                      high_value,
                                                                      predicted_label_path,
                                                                      None,
                                                                      mixed_precision,
                                                                      patch_based_norm,
                                                                      tta = test_time_augmentation,
                                                                      image_dim = training_image_shape,
                                                                      keep_size = keep_size,
                                                                      step_size = step_size,
                                                                      save_files = False)
        validation_target_shapes = [training_source_shapes[index] for index in val_indices]
        # Always resize + export - 2_watershed_tune.py reads these back from disk.
        val_heatmaps = [resize_volume_to_ome_zarr(
            heatmap,
            os.path.join(predicted_label_path, "preds", f"validation_{i}_resized{tag}.ome.zarr"),
            validation_target_shapes[i],
            order=3,
            is_label=False,
        ) for i, heatmap in enumerate(val_heatmaps)]
        val_labels_integer_unpadded = [resize_volume_to_ome_zarr(
            labels_volume,
            os.path.join(predicted_label_path, "preds", f"validation_{i}_labels_resized{tag}.ome.zarr"),
            validation_target_shapes[i],
            order=0,
            is_label=True,
        ) for i, labels_volume in enumerate(val_labels_integer_unpadded)]

    print(f"done! heatmaps written to {os.path.join(predicted_label_path, 'preds')} (tag='{run_id}')")
    print("run 2_watershed_tune.py with the same --config and --run_id to finish segmentation.")

if __name__ == "__main__":
    print("running with seed {}".format(1))

    main(1)
