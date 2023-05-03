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
cd /home/ubuntu/object-detection-tools/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/

# Install the Object Detection API
pip install content/models/research/

# Need to downgrade to TF v2.8.0 due to Colab compatibility bug with TF v2.10 (as of 10/03/22)
pip install tensorflow==2.8.0

# prep directories initially
mkdir directory content/models/mymodel/

# Run Model Bulider Test file, just to verify everything's working properly
python content/models/research/object_detection/builders/model_builder_tf2_test.py