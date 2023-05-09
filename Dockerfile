# this docker image sets up teh environment to run the
# google object detection api

# Use the latest Ubuntu LTS release as the base image
FROM ubuntu:latest

# Update the package lists and upgrade all packages
RUN apt-get update && apt-get upgrade -y

# Install Python 3, Zip, and Git
RUN apt-get install -y python3 zip git

# Install the wget package for Python
RUN apt-get install -y python3-pip
RUN pip3 install wget

# Copy the files from the root directory
COPY ./replace_tensor.py /
COPY ./train_val_test_split.py /
COPY ./setup_variables.py /
COPY ./run_train.sh /
COPY ./grab_data.sh /

# Clone the TensorFlow Lite Object Detection repository and rename the folder
RUN git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git && \
    mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite-odapi

# Move replace_tensor.py script to tflite-odapi directory
RUN mv replace_tensor.py tflite-odapi/

# Set the working directory to tflite-odapi
WORKDIR /tflite-odapi

RUN apt-get install -y protobuf-compiler

# Create content directory and clone TensorFlow models repository
RUN mkdir content && \
    cd content && \
    git clone --depth 1 https://github.com/tensorflow/models && \
    cd models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    cd /tflite-odapi

# Replace TensorFlow with version 2.8.0
RUN python3 replace_tensor.py

# Install the Object Detection API
RUN pip install content/models/research/

# Need to downgrade to TF v2.8.0 due to Colab compatibility bug with TF v2.10 (as of 10/03/22)
RUN pip install tensorflow==2.8.0

# Create necessary directories
RUN mkdir directory content/models/mymodel/

# Run Model Builder Test file, just to verify everything's working properly
RUN python3 content/models/research/object_detection/builders/model_builder_tf2_test.py

# Set the entrypoint to bash, and set the default command to be an empty string
ENTRYPOINT ["/bin/bash"]
CMD []

# Run the setup_env.sh file, validate_data.sh and run_model.sh
CMD ["/bin/bash", "validate_data.sh"]
CMD ["/bin/bash", "run_train.sh"]




# write me a dockerfile that:
# 1. installs the latest ubuntu
# 2. updates and upgrades all packages
# 3. installs python3
# 4. install zip
# 5. installs the python package wget
