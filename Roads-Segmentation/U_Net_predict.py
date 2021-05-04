
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
import keras.callbacks
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as keras
import numpy as np
import cv2 # For CV operations
from PIL import Image  #To create and store images
import h5py
import os

import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Input, Model
import tensorflow.keras as tf

mean = 106.93701;
std = 70.85109;

# load json and create model
json_file = open('StoredModel.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("modelWeights.h5")
print("Loaded model from disk")

loaded_model.save('finalModel.hdf5')
loaded_model = load_model('finalModel.hdf5')

file = h5py.File('Dataset_test.h5', 'r')
imgs_test = file.get('images')
# imgs_mask_test = file.get('masks')
imgs_test = np.array(imgs_test)
# imgs_mask_test = np.array(imgs_mask_test)
imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('*' * 30)
print('Loading saved weights...')
print('*' * 30)
loaded_model.load_weights('modelWeights.h5')

print('*' * 30)
print('Predicting masks on test data...')
print('*' * 30)
imgs_mask_test = loaded_model.predict(imgs_test, verbose=1)

print('*' * 30)
print('Saving predicted masks to files...')
print('*' * 30)
pred_dir = 'PredictionsFromLoadedModel'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for i, image in enumerate(imgs_mask_test):
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_dir, str(i + 1) + '_pred.png'), image)