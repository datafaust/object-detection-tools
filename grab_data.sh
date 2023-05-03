#!/bin/bash  

# grab data -----------------------------------------------



# get images
wget -O content/images.zip https://www.dropbox.com/s/gk57ec3v8dfuwcp/CoinPics_11NOV22.zip?dl=0

# create directories
mkdir content/images
unzip -q content/images.zip -d content/images/all
mkdir content/images/train; mkdir content/images/validation; mkdir content/images/test

# split images
#wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/train_val_test_split.py
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

