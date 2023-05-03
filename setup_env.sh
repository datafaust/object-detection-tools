#!/bin/bash      

# update
sudo apt-get update
sudo apt-get upgrade -y
pip install wget

# install object detection api ------------------------------
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
cd TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
mkdir content
cd content
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
cd /home/ubuntu/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/

# Install the Object Detection API
pip install content/models/research/

# Need to downgrade to TF v2.8.0 due to Colab compatibility bug with TF v2.10 (as of 10/03/22)
pip install tensorflow==2.8.0


# grab data -----------------------------------------------

# prep directories initially
mkdir directory content/models/mymodel/

# Run Model Bulider Test file, just to verify everything's working properly
python content/models/research/object_detection/builders/model_builder_tf2_test.py

# get images
wget -O content/images.zip https://www.dropbox.com/s/gk57ec3v8dfuwcp/CoinPics_11NOV22.zip?dl=0

# create directories
mkdir content/images
unzip -q content/images.zip -d content/images/all
mkdir content/images/train; mkdir content/images/validation; mkdir content/images/test

# split images
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/train_val_test_split.py
python train_val_test_split.py


# create class file and tf records ------------------------------------------
cat <<EOF >> content/labelmap.txt
penny
nickel
dime
quarter
EOF

# tf records and csvs
cd content/
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py
python create_csv.py
python create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord
python create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord

# run configuration ------------------------------------------------------
cd ..
python setup_variables.py

