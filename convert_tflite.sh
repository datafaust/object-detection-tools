#!/bin/bash


# Make a directory to store the trained TFLite model
#output_directory = 'content/custom_model_lite'

# Path to training directory (the conversion script automatically chooses the highest checkpoint file)
#last_model_path = 'content/training'

python content/models/research/object_detection/export_tflite_graph_tf2.py \
    --trained_checkpoint_dir {content/training} \
    --output_directory {content/custom_model_lite} \
    --pipeline_config_path {pipeline_file}