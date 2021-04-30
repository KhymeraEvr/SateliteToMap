import cv2 # For CV operations
from PIL import Image  #To create and store images
import numpy as np

#To binarize the input
import h5py
import os

"""### Mapping the Drive

## Resizing the Images (Input) (Satellite Images)
"""

# Resizing each image (1500 * 1500) to image (256 * 256) and converting the ground truths to binary masks

trainInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\sat'
trainOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\map'
testInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\sat'
testOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\map'
validInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\sat'
validOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\map'

originalImages = os.listdir(trainInputImagesPath)
dim = (256, 256) #(w,h)

for index,image in enumerate(originalImages):
  
  print("Reading Image : " + str(image) +" with Index : "+str(index))
  readImage = cv2.imread("Training/Input/" + str(image), 1)
  
  resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)
  
  imageName = str(image).split(".")[0]
  print("Shape of resized Image is : ", resizedImage.shape)
  
  #Converting to .png and Storing resized image to a directory
  
  cv2.imwrite(os.getcwd() + "/TrainingImages/" + imageName + ".png", resizedImage)
  print("Resized and Stored Image : " + str(image) +" with Index : "+str(index))

"""## Resizing the labels to Binary Masks"""

# Resizing each image (1500 * 1500) to image (256 * 256) and converting the ground truths to binary masks

originalImages = os.listdir("Training/InputLables")
dim = (256, 256) #(w,h)

for index,image in enumerate(originalImages):
  
  print("Reading Image : " + str(image) +" with Index : "+str(index))
  readImage = cv2.imread("Training/InputLables/" + str(image), 0)
  
  resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)
  
  imageName = str(image).split(".")[0]
  
  (thresh, im_bw) = cv2.threshold(resizedImage, 128, 255, cv2.THRESH_BINARY)
  
  print("Shape of resized Image is : ", im_bw.shape)
  
  #Converting to .png and Storing resized image to a directory
  
  cv2.imwrite(os.getcwd() + "/TrainingMasks/" + imageName + ".png", im_bw)
  print("Resized and Stored Image : " + str(image) +" with Index : "+str(index))

"""## Creating input & mask arrays"""

images = []
originalImages = os.listdir("TrainingImages/")

for index,image in enumerate(originalImages):
    print("Image number : " +str(index) )
    img = Image.open("TrainingImages/" + str(image))
    
    arr = np.array(img)
    #arr = np.expand_dims(arr, -1)
    images.append(arr)

masks = []
originalImages = os.listdir("TrainingMasks/")

for index,image in enumerate(originalImages):
  print("Image number : " +str(index) )
  img = Image.open("TrainingMasks/" + str(image))
  arr = np.array(img)
  arr = np.expand_dims(arr, -1)
  masks.append(arr)

images = np.array(images)
masks = np.array(masks)

print(masks.shape)
print(images.shape)

"""## Finalizing Dataset for Training"""

with h5py.File("Dataset_train.h5", 'w') as hdf:
    hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)

"""# U-Net Model for Road Segmentation"""

from keras.models import *
from keras.layers import *
from keras.layers import Conv2D
from keras.optimizers import *
import keras
import keras.callbacks
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as keras

import matplotlib.pyplot as plt

"""## Model Definition"""

def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

"""### Optimizer : Adam, Loss : Binary Cross Entropy"""

print('*'*30)
print('Loading and preprocessing train data...')
print('*'*30)
file = h5py.File('Dataset_train.h5', 'r')
imgs_train = file.get('images')
imgs_mask_train = file.get('masks')
imgs_train = np.array(imgs_train)
imgs_mask_train = np.array(imgs_mask_train)

print(imgs_train.shape)
print(imgs_mask_train.shape)
#imgs_train = imgs_train.reshape(1108,256,256,1)

 
imgs_train = imgs_train.astype('float32')

mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255  # scale masks to [0, 1]

print('*'*30)
print('Creating and compiling model...')
print('*'*30)
model = unet()
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
tensorboard = TensorBoard(log_dir='tensorboard/', write_graph=True, write_images=True)

"""## Model Fitting"""

model.summary()

print('*'*30)
print('Fitting model...')
print('*'*30)
history =  model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=30, verbose=2, shuffle=True,validation_split=0.2,callbacks=[model_checkpoint, tensorboard])

"""## Creating Test Dataset"""

originalImages = os.listdir("Testing/testingInput")
dim = (256, 256) #(w,h)

for index,image in enumerate(originalImages):
  
  print("Reading Image : " + str(image) +" with Index : "+str(index))
  readImage = cv2.imread("Testing/testingInput/" + str(image), 1)
  
  resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)
  
  imageName = str(image).split(".")[0]
  
  #Converting to .png and Storing resized image to a directory
  
  
  cv2.imwrite(os.getcwd() + "/TestingImages/" + imageName + ".png", resizedImage)
  print("Resized and Stored Image : " + str(image) +" with Index : "+str(index))

testImages = []
originalImages = os.listdir("TestingImages/")

for i in originalImages:
    img = Image.open("TestingImages/" + str(i))
    
    arr = np.array(img)
    #arr = np.expand_dims(arr, -1)
    testImages.append(arr)

"""## Finalizing input for testing"""

with h5py.File("Dataset_test.h5", 'w') as hdf:
    hdf.create_dataset('images', data=testImages, compression='gzip', compression_opts=9)

file = h5py.File('Dataset_test.h5', 'r')
imgs_test = file.get('images')
#imgs_mask_test = file.get('masks')
imgs_test = np.array(imgs_test)
#imgs_mask_test = np.array(imgs_mask_test)
imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('*'*30)
print('Loading saved weights...')
print('*'*30)
model.load_weights('weights.h5')

print('*'*30)
print('Predicting masks on test data...')
print('*'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1)

print('*' * 30)
print('Saving predicted masks to files...')
print('*' * 30)
pred_dir = 'Predictions'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for i, image in enumerate(imgs_mask_test):
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_dir, str(i + 1) + '_pred.png'), image)

plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'], linewidth=4, color='r')                   #visualising training and validation loss curves
plt.plot(history.history['val_loss'], linewidth=4, color='b')
plt.title('Model train vs Validation Loss', fontsize=20, fontweight="bold")
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Validation'], loc='upper right', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

"""# Model Saving"""

# keras library import  for Saving and loading model and weights

from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = model.to_json()


with open("StoredModel.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("modelWeights.h5")

"""## Using the stored Model to predict & Test"""

# load json and create model
json_file = open('StoredModel.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("modelWeights.h5")
print("Loaded model from disk")

loaded_model.save('finalModel.hdf5')
loaded_model=load_model('finalModel.hdf5')

file = h5py.File('Dataset_test.h5', 'r')
imgs_test = file.get('images')
#imgs_mask_test = file.get('masks')
imgs_test = np.array(imgs_test)
#imgs_mask_test = np.array(imgs_mask_test)
imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('*'*30)
print('Loading saved weights...')
print('*'*30)
loaded_model.load_weights('modelWeights.h5')

print('*'*30)
print('Predicting masks on test data...')
print('*'*30)
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