import cv2 # For CV operations
from PIL import Image  #To create and store images
import numpy as np
from keras.models import *

#To binarize the input
import h5py
import os

predictInputPath = 'Predictions/input/'
resizedFolder = 'Predictions/resized/'
outputFolder = 'Predictions/output/'
dim = (256, 256) #(w,h)
mean = 106.93701;
std = 70.85109;


def Predict(fileName):
    #for index,image in enumerate(originalImages): resize training input to png
    #
    print("Reading Image : " + fileName)
    readImage = cv2.imread(predictInputPath + fileName, 1)

    resizedImage = cv2.resize(readImage, dim, interpolation=cv2.INTER_AREA)

    imageName = str(fileName).split(".")[0]
    print("Shape of resized Image is : ", resizedImage.shape)

    #Converting to .png and Storing resized image to a directory

    cv2.imwrite(resizedFolder + imageName + ".png", resizedImage)
    print("Resized and Stored Image : " + str(imageName))
    inputArray = [];
    arr = np.array(resizedImage)
    inputArray.append(arr)
    inputArray = np.array(inputArray)
    #inputArray = np.array(imgs_test)
    # imgs_mask_test = np.array(imgs_mask_test)
    imgs_predict = inputArray.astype('float32')
    imgs_predict -= mean
    imgs_predict /= std

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

    print('*' * 30)
    print('Loading saved weights...')
    print('*' * 30)
    loaded_model.load_weights('modelWeights.h5')

    print('*' * 30)
    print('Predicting masks on test data...')
    print('*' * 30)
    imgs_mask_test = loaded_model.predict(imgs_predict, verbose=1)

    print('*' * 30)
    print('Saving predicted masks to files...')
    print('*' * 30)
    pred_dir = outputFolder
    resultFilename = ''
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for i, image in enumerate(imgs_mask_test):
        image = (image * 255).astype(np.uint8)
        resultFilename = imageName + '_pred.png'
        cv2.imwrite(os.path.join(pred_dir, resultFilename), image)

    return resultFilename

#Predict('Screenshot_7.png')



