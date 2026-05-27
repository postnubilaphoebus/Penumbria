import torch
import numpy as np
import torch
from utils import load_training_images_and_labels, preprocess_0_1, load_inference_images
import os
from train import train_model
from inference import sliding_window_inference
from postprocess import watershed_inference, objective
from skimage.transform import resize
import warnings
import os
import tifffile
import torch
from typing import Dict
import yaml
import sys
import copy
import random
from datetime import datetime
import optuna
import argparse
from U_VixLSTM.UVixLSTM import UVixLSTM

def load_config(config_path: str) -> Dict:
    """loads the yaml config file

    Args:
        config_path (str): _description_

    Returns:
        Dict: _description_
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def override_config(config, args):
    """Apply CLI argument overrides to nested config dictionary
    
    Args:
        config (dict): Nested configuration dictionary
        args (Namespace): Parsed command-line arguments
    
    Returns:
        dict: Configuration with CLI overrides applied
    """
    merged_config = copy.deepcopy(config)
    
    for key, value in vars(args).items():
        if value is None or key == 'config':
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
    # For example: python 2_segment.py -c="./dataset_configs/zebrafish_confocal.yaml"
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
        description='Heatmap neural network for cell segmentation'
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
    parser.add_argument('--inference_resolution_upsampling', type=float,
        help='If set, upsample resolution during inference (e.g., 2.0)')

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
    # postprocessing section

    parser.add_argument('--parameter_tuning', type=int,
        help='whether to perform parameter tuning on validation data')
    parser.add_argument('--cell_prominence', type=float,
        help='Minimum prominence to detect a cell (watershed threshold)')
    parser.add_argument('--cell_confidence_minimum', type=float,
        help='Minimum heatmap confidence for cells')
    parser.add_argument('--background_threshold', type=float,
        help='Threshold for separating background from cells')
    parser.add_argument('--minimum_cell_size', type=int,
        help='Cells smaller than this (in pixels) will be removed')
    parser.add_argument('--gaussian_smoothing', type=bool,
        help='Apply Gaussian smoothing before segmentation')
    parser.add_argument('--simple_thresholding', type=bool,
        help='Use simple (non-learned) thresholding for segmentation')

    args = parser.parse_args()
    args_dict = vars(args)
    config_path = args_dict.get("config")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    # Load base config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    # Apply CLI overrides
    merged_config = override_config(config, args)

    # Extract configuration sections
    label_cfg = merged_config['label_transform']
    model_cfg = merged_config['model']
    train_cfg = merged_config['training']
    inference_cfg = merged_config['inference']
    post_cfg = merged_config['postprocessing']

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
    use_sgg_layer = model_cfg['use_sgg_layer']

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

    # Postprocessing
    cell_prominence = post_cfg['cell_prominence']
    cell_confidence_minimum = post_cfg['cell_confidence_minimum']
    background_threshold = post_cfg['background_threshold']
    minimum_cell_size = post_cfg['minimum_cell_size']
    gaussian_smoothing = post_cfg['gaussian_smoothing']
    simple_thresholding = post_cfg['simple_thresholding']
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
    
    resizing_necessary = [1.0 != f for f in resizing_factors]
    resizing_factors = [int(f) if f == 1.0 else f for f in resizing_factors]
    resizing_necessary = np.array(resizing_necessary)
    
    if np.any(resizing_necessary):
        print("resizing necessary")
        print("resizing factors:", resizing_factors)
    else:
        print("resizing not necessary")

    print("preprocessing training and inference images ...")
    images = [preprocess_0_1(image, low_clip=1.0, high_clip=99.9) for image in images]

    if inference_path is None:
        inference_images = [images[idx] for idx in inference_indices]
        inference_filenames = [str(idx) for idx in inference_indices]
        current_path = os.getcwd()
        project_path_inference = os.path.join(current_path, "inference_data")
        predicted_label_path = project_path_inference

    else:
        inference_images, inference_filenames = load_inference_images(inference_path, fileformat=".tif")
        if dynamic_cropping:
            inference_images = [preprocess_0_1(image, low_clip=1.0, high_clip=99.9) for image in inference_images]
        inference_indices = [-10000000, -20000000]


    print("inference filenames", inference_filenames)
    assert inference_images[0].min() == 0.0 and inference_images[0].max() == 1.0, \
           "images must be normalized between 0 and 1"
    assert images[0].min() == 0.0 and images[0].max() == 1.0, "images must be normalized between 0 and 1"

    mask_file_matrix = []
    mask_filename_matrix = []

    if inference_resolution_upsampling is not None:
        print("resizing inference images...")
        print("this may take a while for higher order spline interpolation (e.g. 3)")
        inference_images = [resize(img, (img.shape[0] * inference_resolution_upsampling[0], 
                                         img.shape[1] * inference_resolution_upsampling[1], 
                                         img.shape[2] * inference_resolution_upsampling[2]), order = 3) \
                                         for img in inference_images]
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels_intensity = labels

    if dynamic_cropping:
        apply_padding = True  
    else:
        apply_padding = False

    print("pad images for training (if dynamic cropping enabled)...")

    # Validate dimensions
    if len(training_image_shape) not in [2, 3]:
        raise ValueError(f"Incorrect training image shape provided: {training_image_shape}")

    # Calculate padding
    maximum_training_image_dim = max(training_image_shape)
    pad_length = maximum_training_image_dim if apply_padding else int(maximum_training_image_dim / 2)

    # Store unpadded copies
    images_unpadded = images.copy()
    integer_labels_unpadded = integer_labels.copy()

    if apply_padding:
        # Determine padding pattern based on dimensionality
        pad_width = ((pad_length, pad_length),) * len(training_image_shape)
        
        # Apply padding
        images = [np.pad(img, pad_width, "reflect") for img in images]
        labels_intensity = [np.pad(img, pad_width, 'reflect') for img in labels_intensity]
        integer_labels = [np.pad(img, pad_width, 'constant', constant_values=-100) 
                        for img in integer_labels]
    else:
        # Keep unmodified arrays
        images = images.copy()
        labels_intensity = labels_intensity.copy()
        integer_labels = integer_labels.copy()

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
        print("use_sgg_layer: ", use_sgg_layer)
        model = UVixLSTM(class_num=1, 
                         img_dim=training_image_shape[0], 
                         out_channels=64, 
                         depth = 12, 
                         dim = 256,
                         use_sgg_layer = use_sgg_layer).to(device)
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
                            use_sgg_layer = use_sgg_layer)

        seed_string = str(seed)
        model_weights_path = seed_string + "_" + model_weights_path
        torch.save(model.state_dict(), model_weights_path)

    else:
        print("loading model weights...")
        model = UVixLSTM(class_num=in_channels, 
                         img_dim=training_image_shape[0], 
                         out_channels = 64, 
                         depth = 12, 
                         dim = 256).to(device)
        try:
            model.load_state_dict(torch.load(model_weights_path))
        except:
            raise Exception("model weights not found at {}".format(model_weights_path))

    ####################################################################################################################
    ##################################### model inference ##############################################################
    ####################################################################################################################

    if dynamic_cropping:
        patch_based_norm = False
    else:
        patch_based_norm = True

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

    if np.any(resizing_necessary):
        print("resizing...")
        padding_list_inferece = [None] * len(model_prediction)
        model_prediction = [resize(model_prediction[i], (model_prediction[i].shape[0] / resizing_factors[0], 
                                                         model_prediction[i].shape[1] / resizing_factors[1], 
                                                         model_prediction[i].shape[2] / resizing_factors[2]), 
                                                         anti_aliasing=True, order = 3) \
                                                         for i in range(len(model_prediction))]
    
    ####################################################################################################################
    ##################################### watershed tuning #############################################################
    ####################################################################################################################

    if parameter_tuning:

        print("watershed tuning...")
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
        if np.any(resizing_necessary):
            val_heatmaps = [resize(val_heatmaps[i], (val_heatmaps[i].shape[0] / resizing_factors[0], 
                                                     val_heatmaps[i].shape[1] / resizing_factors[1], 
                                                     val_heatmaps[i].shape[2] / resizing_factors[2]), 
                                                     anti_aliasing=True, order = 3) for i in range(len(val_heatmaps))]
            padding_list_val = [None] * len(val_heatmaps)
            val_labels_integer_unpadded = [resize(img1_resized, (img1_resized.shape[0] / resizing_factors[0], 
                                                                 img1_resized.shape[1] / resizing_factors[1], 
                                                                 img1_resized.shape[2] / resizing_factors[2]), 
                                                                 anti_aliasing=False, order = 0) \
                                                                    for img1_resized in val_labels_integer_unpadded]

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, 
                                               val_heatmaps, 
                                               val_labels_integer_unpadded, 
                                               padding_list_val, 
                                               data_dimensionality), n_trials=500)

        besth = study.best_params['h']
        best_cc = study.best_params['c']
        best_bg = study.best_params['bg']
        best_gaussian = study.best_params['gaussian_smoothing']
        best_thresh = study.best_params['simple_thresholding']

        print("best parameters: h = {}, c = {}, bg = {}, gaussian_smoothing = {}, simple_thresholding = {},".\
              format(besth, 
                     best_cc, 
                     best_bg, 
                     best_gaussian, 
                     simple_thresholding))
        
        print(f"best map on validation set: {study.best_value:.5f}")
                                                                                                                            

    else:

        besth = cell_prominence
        best_cc = cell_confidence_minimum
        best_bg = background_threshold
        best_gaussian = gaussian_smoothing
        best_thresh = simple_thresholding
    
    ###################################################################################################################
    ##################################### watershed flooding ##########################################################
    ###################################################################################################################

    predicted_label_path = os.path.join(predicted_label_path, "preds")
    if not os.path.exists(predicted_label_path):
        os.makedirs(predicted_label_path)
    
    print("neural network done, starting watershed postprocessing...")
    for inference_filename, prediction, padd in zip(inference_filenames, model_prediction, padding_list_inferece):

        wts = watershed_inference(prediction,
                                  padding = padd,
                                  minimum_cell_size = minimum_cell_size,
                                  h = besth,
                                  cell_confidence_minimum = best_cc,
                                  background_threshold = best_bg,
                                  gaussian_smoothing = best_gaussian,
                                  simple_thresholding=best_thresh,
                                  low_confidence_merging = False,
                                  sym = False)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(predicted_label_path, f"{inference_filename}_labels_predicted_{timestamp}.tif")
        tifffile.imwrite(filepath, wts.astype(np.float32))
        print("file {filepath} saved, number of cells = {num_features}".format(filepath = filepath, 
                                                                               num_features = len(np.unique(wts)) - 1))

    print("done!")

if __name__ == "__main__":
    print("running with seed {}".format(0)) 

    main(0)



