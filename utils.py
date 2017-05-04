# some code from https://github.com/carpedm20/DCGAN-tensorflow
import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import cv2

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def read_image(img_path grayscale=False):
    img = cv2.imread(img_path)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def save_image(img, filename, path):
    cv2.imwrite(str(path) + str(filename), img)

def transform(img, out_h, out_w):
    img = cv2.imresize(img, out_w)
