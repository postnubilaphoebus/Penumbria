import numpy as np
import logging
import os
import glob
import copy
import sys
import argparse
import yaml
from typing import Dict
from datetime import datetime
import optuna
from postprocess import watershed_inference_auto, objective, prepare_tuning_patches
from volume_io import (
    ChunkedVolume,
    export_zarr_to_tiff,
    imagej_label_dtype,
    open_ome_zarr,
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


def main():

    ###################################################################################################################
    ######################################### configuration and initialization ########################################
    ###################################################################################################################

    # Run this AFTER 1_train_and_infer.py, with the SAME -c config and SAME --run_id,
    # so it can find that run's validation/inference heatmaps.
    # Example: python 2_watershed_tune.py -c="./dataset_configs/zebrafish_confocal.yaml" --run_id="run1"

    parser = argparse.ArgumentParser(
        prog='Penumbria',
        description='Watershed parameter tuning + final label prediction'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default = "default_config.yaml",
        help='Path to YAML configuration file'
    )

    parser.add_argument('--run_id', type=str, default="",
        help='Identifier used by 1_train_and_infer.py for this run (e.g. "run1")')

    # ───────────────────────────────────────────────────────────────
    # training section (only what's needed to locate this run's heatmaps)

    parser.add_argument('--data_dimensionality', type=int,
        help='Data dimensionality: 2D or 3D')
    parser.add_argument('--val_indices', type=int, nargs='+',
        help='Indices of validation images')
    parser.add_argument('--training_image_shape', type=int, nargs='+',
        help='Training input patch size (e.g., 64 64 64)')
    parser.add_argument('--inference_folder', type=str,
        help='Where images to segment are located')

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
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    merged_config = override_config(config, args)

    train_cfg = merged_config['training']
    inference_cfg = merged_config['inference']
    post_cfg = merged_config['postprocessing']

    # run identifier: must match the one passed to 1_train_and_infer.py
    run_id = args.run_id
    tag = f"_{run_id}" if run_id else ""

    data_dimensionality = train_cfg['data_dimensionality']
    training_image_shape = train_cfg['training_image_shape']
    val_indices = train_cfg['val_indices']
    inference_path = train_cfg['inference_folder']

    inference_indices = inference_cfg['inference_indices']

    cell_prominence = post_cfg['cell_prominence']
    cell_confidence_minimum = post_cfg['cell_confidence_minimum']
    background_threshold = post_cfg['background_threshold']
    minimum_cell_size = post_cfg['minimum_cell_size']
    gaussian_smoothing = post_cfg['gaussian_smoothing']
    simple_thresholding = post_cfg['simple_thresholding']
    parameter_tuning = post_cfg['parameter_tuning']

    # Same predicted_label_path resolution as 1_train_and_infer.py
    external_inference = inference_path is not None
    if not external_inference:
        predicted_label_path = os.path.join(os.getcwd(), "inference_data")
    else:
        predicted_label_path = inference_path

    preds_dir = os.path.join(predicted_label_path, "preds")
    if not os.path.exists(preds_dir):
        raise FileNotFoundError(
            f"{preds_dir} does not exist. Run 1_train_and_infer.py first with the same "
            f"--config and --run_id."
        )

    ###################################################################################################################
    ##################################### load this run's heatmaps from disk ##########################################
    ###################################################################################################################

    suffix = f"_inference_output_resized{tag}.ome.zarr"
    inference_heatmap_paths = sorted(glob.glob(
        os.path.join(preds_dir, f"*{suffix}")
    ))
    if not inference_heatmap_paths:
        raise FileNotFoundError(f"No *{suffix} files found in {preds_dir}")
    inference_filenames = [os.path.basename(p)[:-len(suffix)] for p in inference_heatmap_paths]
    model_prediction = [open_ome_zarr(p) for p in inference_heatmap_paths]
    padding_list_inferece = [None] * len(model_prediction)

    print("found inference heatmaps:", inference_filenames)

    ####################################################################################################################
    ##################################### watershed tuning #############################################################
    ####################################################################################################################

    if parameter_tuning:

        print("watershed tuning...")
        val_heatmaps = [open_ome_zarr(os.path.join(preds_dir, f"validation_{i}_resized{tag}.ome.zarr"))
                         for i in range(len(val_indices))]
        val_labels_integer_unpadded = [open_ome_zarr(os.path.join(preds_dir, f"validation_{i}_labels_resized{tag}.ome.zarr"))
                                        for i in range(len(val_indices))]
        padding_list_val = [None] * len(val_heatmaps)

        if any(isinstance(item, ChunkedVolume) for item in val_heatmaps):
            print("parameter tuning uses representative out-of-core validation regions")
            val_heatmaps, val_labels_integer_unpadded = prepare_tuning_patches(
                val_heatmaps,
                val_labels_integer_unpadded,
                val_heatmaps,
                training_image_shape,
            )
            padding_list_val = [None] * len(val_heatmaps)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial,
                                               val_heatmaps,
                                               val_labels_integer_unpadded,
                                               padding_list_val,
                                               data_dimensionality), n_trials=300)

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
                     best_thresh))

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

    print("starting watershed postprocessing on inference heatmaps...")
    for inference_filename, prediction, padd in zip(inference_filenames, model_prediction, padding_list_inferece):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        zarr_path = os.path.join(
            preds_dir, f"{inference_filename}_labels_predicted_{timestamp}{tag}.ome.zarr"
        )
        wts, num_features = watershed_inference_auto(
            prediction,
            zarr_path,
            minimum_cell_size=minimum_cell_size,
            h=besth,
            cell_confidence_minimum=best_cc,
            background_threshold=best_bg,
            gaussian_smoothing=best_gaussian,
            simple_thresholding=best_thresh,
        )

        # OME-Zarr is the working output. Stream an exact, native ImageJ label
        # stack as uint16, or float32 if more than 65535 labels are present.
        filepath = os.path.join(preds_dir, f"{inference_filename}_labels_predicted_{timestamp}{tag}.tif")
        export_zarr_to_tiff(wts, filepath, output_dtype=imagej_label_dtype(num_features))
        print("file {filepath} saved, number of cells = {num_features}".format(filepath = filepath,
                                                                               num_features = num_features))

    print("done!")

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    logging.getLogger("penumbria.gpu").setLevel(logging.INFO)  # GPU memory budget and patch plan
    main()
