#!/bin/bash  

# copy s3 contents
aws s3 cp s3://odapi-1/model_inputs/sample_model.zip .

# unzip
unzip -d . sample_model.zip

# create directories
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/custom_model_lite
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images/all
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images/train
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images/validation
mkdir TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images/test


# copy data over to respective folders
cp  -r sample_model/images TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/images/all
cp  -r sample_model/labelmap.txt TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/content/

# move to working directory and run data
cd TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
python train_val_test_split.py

# create tf records and csvs
cd content/
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py
python create_csv.py
python create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord
python create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord


# run setup
cd ..
python setup_variables.py

# launch tensor board


# run train
python content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=content/models/mymodel/pipeline_file.config \
    --model_dir=content/training/ \
    --alsologtostderr \
    --num_train_steps=5000 \
    --sample_1_of_n_eval_examples=1

# create tflite
python content/models/research/object_detection/export_tflite_graph_tf2.py \
    --trained_checkpoint_dir=content/training/ \
    --output_directory=content/custom_model_lite/ \
    --pipeline_config_path=content/models/mymodel/pipeline_file.config

python tflite_convert.py

# run map-------------------------------------
git clone https://github.com/Cartucho/mAP content/mAP
cd content/mAP
rm input/detection-results/* 
rm input/ground-truth/* 
rm input/images-optional/* 
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/calculate_map_cartucho.py

cd ../../
cp content/images/test/* content/mAP/input/images-optional # Copy images and xml files
mv content/mAP/input/images-optional/*.xml content/mAP/input/ground-truth/  # Move xml files to the appropriate folder

python content/mAP/scripts/extra/convert_gt_xml.py

python run_map.py

mv content/labelmap.txt content/mAP
cd content/mAP
python calculate_map_cartucho.py --labels=labelmap.txt

# write tfite and results set to s3 -------------------
cd ../../
# Move labelmap and pipeline config files into TFLite model folder and zip it up
cp content/labelmap.txt content/custom_model_lite
cp content/labelmap.pbtxt content/custom_model_lite
cp content/models/mymodel/pipeline_file.config content/custom_model_lite

cd content
zip -r custom_model_lite.zip custom_model_lite


aws s3 cp custom_model_lite.zip s3://odapi-1/model_outputs/