import cv2 # For CV operations
from PIL import Image  #To create and store images
import numpy as np

#To binarize the input
import h5py
import os

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.python.keras.optimizers import Adam

"""### Mapping the Drive

## Resizing the Images (Input) (Satellite Images)
"""

# Resizing each image (1500 * 1500) to image (256 * 256) and converting the ground truths to binary masks

print(os.getcwd())

trainInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\sat'
trainOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\map'
testInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\sat'
testOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\map'
validInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\sat'
validOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\map'
outputFolder = 'E:\\Datasets\\SateliteToMaps\\dataset\\TrainingImages\\'
outputFolderMasks = 'E:\\Datasets\\SateliteToMaps\\dataset\\TrainingMasks\\'
outputTestFolder = 'E:\\Datasets\\SateliteToMaps\\dataset\\TestingImages\\'

originalImages = os.listdir(trainInputImagesPath)
dim = (256, 256) #(w,h)

#for index,image in enumerate(originalImages): resize training input to png
#
#  print("Reading Image : " + str(image) +" with Index : "+str(index))
#  readImage = cv2.imread(trainInputImagesPath +'\\' + str(image), 1)
#
#  resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)
#
#  imageName = str(image).split(".")[0]
#  print("Shape of resized Image is : ", resizedImage.shape)
#
#  #Converting to .png and Storing resized image to a directory
#
#  cv2.imwrite(outputFolder + imageName + ".png", resizedImage)
#  print("Resized and Stored Image : " + str(image) +" with Index : "+str(index))
#
#"""## Resizing the labels to Binary Masks"""

# Resizing each image (1500 * 1500) to image (256 * 256) and converting the ground truths to binary masks

originalImages = os.listdir(trainOutputImagesPath)
dim = (256, 256) #(w,h)

for index,image in enumerate(originalImages):
  
  print("Reading Image : " + str(image) +" with Index : "+str(index))
  readImage = cv2.imread(trainOutputImagesPath + '\\' + str(image), 0)
  
  resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)
  
  imageName = str(image).split(".")[0]
  
  (thresh, im_bw) = cv2.threshold(resizedImage, 128, 255, cv2.THRESH_BINARY)
  
  print("Shape of resized Image is : ", im_bw.shape)
  
  #Converting to .png and Storing resized image to a directory
  
  cv2.imwrite(outputFolderMasks  +  imageName + ".png", im_bw)
  print("Resized and Stored Image : " + str(image) +" with Index : "+str(index))

"""## Creating input & mask arrays"""

images = []
originalImages = os.listdir(outputFolder)

for index,image in enumerate(originalImages):
    print("Image number : " +str(index) )
    img = Image.open(outputFolder + str(image))
    
    arr = np.array(img)
    #arr = np.expand_dims(arr, -1)
    images.append(arr)

masks = []
originalImages = os.listdir(outputFolderMasks)

for index,image in enumerate(originalImages):
  print("Image number : " +str(index) )
  img = Image.open(outputFolderMasks + str(image))
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

