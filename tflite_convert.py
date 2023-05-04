# Convert exported graph file into TFLite model file
import tensorflow as tf
import os

home_dir = '/home/ubuntu/object-detection-tools/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/'

os.chdir(home_dir)

converter = tf.lite.TFLiteConverter.from_saved_model('content/custom_model_lite/saved_model')
tflite_model = converter.convert()

with open(os.getcwd() + '/' + 'content/custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)