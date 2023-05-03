#!/bin/bash 
# run final script
python content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=content/models/mymodel/pipeline_file.config \
    --model_dir=content/training/ \
    --alsologtostderr \
    --num_train_steps=40000 \
    --sample_1_of_n_eval_examples=1