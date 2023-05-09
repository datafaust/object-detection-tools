#!/bin/bash  

# copy s3 contents
aws s3 cp s3://odapi-input/model.zip .

# unzip
unzip -d . model.zip

# does the directory contain the pertinent files?
if [ -d "model/images" ] && [ -f "project/data/labelmap.txt" ]; then
    echo "Directory 'images' exists and 'labelmap.txt' file exists inside 'project' directory"
else
    echo "Directory 'images' does not exist or 'labelmap.txt' file does not exist inside 'project' directory"
fi
