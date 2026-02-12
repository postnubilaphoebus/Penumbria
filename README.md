Penumbria

Precision 3D Instance Segmentation via Heatmap Learning

Penumbria is a deep learning framework for high-precision 3D instance segmentation of volumetric data.
Instead of predicting binary masks directly, the network learns to predict Euclidean distance-based heatmaps, which are then converted into instance segmentations using morphological reconstruction and watershed flooding.

The method is particularly effective for separating touching objects in dense 3D volumes.

Overview

The workflow consists of three stages:

Prepare training data

Train the heatmap network

Postprocess predictions into instances

Installation

Create a fresh Conda environment:

conda create -n penumbria python=3.10
conda activate penumbria
pip install -r requirements.txt


Install PyTorch separately:

pip install torch torchvision


Penumbria works with a wide range of PyTorch versions (tested with 2.5.1).

Step 1 — Prepare Training Data

Run:

python 1_prepare_training_data.py --base_path <path> --dataset_name <name>

Arguments
--resizing_factors        Resizing factors for up/downsampling
                          Example: --resizing_factors=7,1,1
                          Default: [1.0, 1.0, 1.0]

--base_path               Base path to images and masks

--output_path             Output folder (default: "prepped_data")

--dataset_name            Dataset name

--img_filter              Image filter string (default: "img")

What This Step Does

Organizes images and labels into a structured dataset folder

Computes Euclidean Distance Transforms (EDT) from labels

Saves these as target heatmaps

Stores resizing factors (if used)

The generated heatmaps are the regression targets for the network.

Step 2 — Training and Segmentation

Run:

python 2_segment.py --config your_config.yaml


Configuration is handled through a YAML file.

YAML Configuration

Below is a representative configuration structure:

label_transform:
  high_value: 20.0
  low_value: -20.0
  background_maximum: -5.0
  foreground_minimum: -2.0
  ignore_index: -100

model:
  optimizer: sgd
  learning_rate: 0.001
  load_pretrained: false
  momentum: 0.9
  model_weights_path: "best_model.pth"

training:
  training_iterations: 6000
  evaluation_interval: 20
  in_channels: 1
  data_dimensionality: 3
  mixed_precision: true
  dynamic_cropping: false
  training_image_shape: [64, 64, 64]
  val_indices: [0, 4]
  verbosity_flag: true
  data_augmentation_types:
    - rotate
    - motion_blur
    - gaussian_noise
  mini_batch_size: 1
  early_stopping_patience: 8000
  training_folder: "datasets/zebrafish_euclid"
  inference_folder: null
  inference_resolution_upsampling: null

inference:
  test_time_augmentation: true
  keep_size: [32, 32, 32]
  step_size: [32, 32, 32]
  inference_indices: [1]

postprocessing:
  parameter_tuning: false
  cell_prominence: 0.23
  cell_confidence_minimum: 0.51
  background_threshold: 0.06
  minimum_cell_size: 9
  gaussian_smoothing: true
  simple_thresholding: false

Sections That Should Not Be Modified
label_transform

Do not change this section.
These values are already defined during data preparation and are only read by the script for consistency.

model

Only modify this section if:

You want to load pretrained weights

You explicitly want to change optimizer settings

Otherwise, leave it unchanged.

Training Guidelines
Training Iterations

If training on full images (no subsampling):

Rule of thumb:
~1500 iterations per training image
(total images − validation − test)

Example:
4 training images → ~6000 iterations

For larger datasets or large volumes:

Use ~100,000 iterations

Large Volumes and Dynamic Cropping

Maximum input size is roughly:

~192³ (theoretical)

128³ (safer)

For larger images, enable:

dynamic_cropping: true


Dynamic cropping randomly samples subvolumes during training and is recommended for large 3D data.

training_image_shape

Defines the patch size used for training.

Example:

training_image_shape: [64, 64, 64]


Increase cautiously depending on GPU memory.

Inference
Test-Time Augmentation (TTA)
test_time_augmentation: true


Keep enabled if using a GPU and moderate dataset size.

Disable if running on CPU or processing very large datasets.

Performance differences are usually modest.

Sliding Window Inference

keep_size and step_size define the sliding window parameters.

Best practice:
Set both to half of training_image_shape.

Example:

training_image_shape: [64,64,64]
keep_size: [32,32,32]
step_size: [32,32,32]


Predictions are merged using Euclidean distance feathering.

inference_folder

Leave as null during cross-validation.

Set to a folder path for explicit test-set inference.

inference_resolution_upsampling

Only use this if test images have a different resolution than training images.

Example:

inference_resolution_upsampling: [1,1,2]


Otherwise leave as null.

Postprocessing

After predicting heatmaps, Penumbria performs:

Optional Gaussian smoothing

Seed detection using h-dome transform

Morphological reconstruction

Watershed flooding

Confidence-based filtering

Postprocessing Parameters

These can be fine-tuned or left at preset values.

Key Parameters

cell_prominence
The h parameter in the h-dome transform. Controls seed strength.

cell_confidence_minimum
Minimum confidence required to retain a detected object.

background_threshold
Separates foreground from background.

minimum_cell_size
Removes small artifacts.

gaussian_smoothing
Smooth heatmap before reconstruction (recommended).

simple_thresholding
Bypasses morphological reconstruction and applies direct thresholding.
Use only if speed is critical.

Recommended Workflow

Prepare data

Train the network

Run on validation data

Adjust postprocessing thresholds

Run final inference

In most cases, only the following fields need modification:

training_folder

inference_folder

training_iterations

inference_resolution_upsampling (if necessary)

All other parameters are already tuned for stable performance.

Summary

Penumbria performs 3D instance segmentation by:

Learning Euclidean distance heatmaps

Detecting object centers robustly

Applying morphological reconstruction

Separating instances with watershed flooding

This approach provides stable separation of touching objects and reliable segmentation in dense volumetric datasets.
