# -*- coding: utf-8 -*-
"""PerPixelRoadSegmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wqc4NMiV9atOkxtD8Pyhj3pijByR-i0T
"""

import random
from os import listdir
from PIL import Image
import os
from datetime import datetime
import tensorflow as tf

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/My Drive/Colab Notebooks/cropped_months/dataset/

"""## Paths to Training, Testing & Validation"""

trainInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\sat'
trainOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\train\\map'
testInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\sat'
testOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\test\\map'
validInputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\sat'
validOutputImagesPath = 'E:\\Datasets\\SateliteToMaps\\dataset\\valid\\map'

trainInputImagesFiles = listdir(trainInputImagesPath)
trainOutputImagesFiles = listdir(trainOutputImagesPath)
testInputImagesFiles = listdir(testInputImagesPath)
testOutputImagesFiles = listdir(testOutputImagesPath)
validInputImagesFiles = listdir(validInputImagesPath)
validOutputImagesFiles = listdir(validOutputImagesPath)

"""### Number of input vs lables Check"""

print(str(datetime.now()) + ': trainInputImagesFiles:', len(trainInputImagesFiles))
print(str(datetime.now()) + ': trainOutputImagesFiles:',  len(trainOutputImagesFiles))
if(len(trainInputImagesFiles) != len(trainOutputImagesFiles)):
    raise Exception('train input images and output images number mismatch')

print(str(datetime.now()) + ': testInputImagesFiles:', len(testInputImagesFiles))
print(str(datetime.now()) + ': testOutputImagesFiles:', len(testOutputImagesFiles))
if(len(testInputImagesFiles) != len(testOutputImagesFiles)):
    raise Exception('test input images and output images number mismatch')

print(str(datetime.now()) + ': validInputImagesFiles:', len(validInputImagesFiles))
print(str(datetime.now()) + ': validOutputImagesFiles:', len(validOutputImagesFiles))
if(len(validInputImagesFiles) != len(validOutputImagesFiles)):
    raise Exception('valid input images and output images number mismatch')

"""## IF Mismatch, Index Check"""

for i in range(len(trainInputImagesFiles)):
    inputImageFile = trainInputImagesFiles[i][:-5]
    outputImageFile = trainOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('train inputImageFile and outputImageFile mismatch at index', str(i))

for i in range(len(testInputImagesFiles)):
    inputImageFile = testInputImagesFiles[i][:-5]
    outputImageFile = testOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('test inputImageFile and outputImageFile mismatch at index', str(i))
        
for i in range(len(validInputImagesFiles)):
    inputImageFile = validInputImagesFiles[i][:-5]
    outputImageFile = validOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('valid inputImageFile and outputImageFile mismatch at index', str(i))

print(str(datetime.now()) + ': input and output files check success')

"""## Writing to Files Subroutine"""

def writeDataFile(inputImagePath, outputImagePath, inputImageFiles, outputImageFiles, dataFileName):
    dataFile = open(dataFileName, 'w')
    rectSize = 5
    linesCount = 0
    linesLimit = 200000
    linesCountPerImage = 0
    linesLimitPerImage = (linesLimit / len(inputImageFiles)) + 1
    
    for i in range(len(inputImageFiles)):
        print(str(datetime.now()) + ': prcessing image', i)
        linesCountPerImage = 0
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size
        # inputImagePixels = inputImage.load()
        
        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()
        
        if((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('train inputImage and outputImage mismatch at index', str(i))

        outputImageRoadPixelsArr = [];
        outputImageNonRoadPixelsArr= [];
        
        for x in range(rectSize//2, inputImageXSize - (rectSize//2)):
            for y in range(rectSize//2, inputImageYSize - (rectSize//2)):
                isRoadPixel = outputImagePixels[x, y]
                if(isRoadPixel):
                    outputImageRoadPixelsArr.append((x, y))
                else:
                    outputImageNonRoadPixelsArr.append((x, y))

        random.shuffle(outputImageRoadPixelsArr)
        random.shuffle(outputImageNonRoadPixelsArr)
        
        for m in range(len(outputImageRoadPixelsArr)):
            if(linesCountPerImage >= linesLimitPerImage):
                break
            
            if(((m*2) + 1) >= len(outputImageNonRoadPixelsArr)):
                break
            
            x = outputImageRoadPixelsArr[m][0];
            y = outputImageRoadPixelsArr[m][1];
            
            rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
            subImage = inputImage.crop(rect).load()
            line = ''
            for i in range(rectSize):
                for j in range(rectSize):
                    line += str(subImage[i, j][0]) + ','
                    line += str(subImage[i, j][1]) + ','
                    line += str(subImage[i, j][2]) + ','
            
            line += str(1) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)
            
            for n in range(2):
                x = outputImageNonRoadPixelsArr[(m*2) + n][0];
                y = outputImageNonRoadPixelsArr[(m*2) + n][1];
                
                rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
                subImage = inputImage.crop(rect).load()
                line = ''
                for i in range(rectSize):
                    for j in range(rectSize):
                        line += str(subImage[i, j][0]) + ','
                        line += str(subImage[i, j][1]) + ','
                        line += str(subImage[i, j][2]) + ','
                
                line += str(0) + '\n'
                linesCount += 1
                linesCountPerImage += 1
                dataFile.write(line)
    
    print(str(datetime.now()) + ': ' + dataFileName + ' linesCount:', linesCount)

"""## Creating Files"""

trainDataFileName = 'dataset/train.csv'
testDataFileName = 'dataset/test.csv'
validDataFileName = 'dataset/valid.csv'

print(str(datetime.now()) + ': writing trainDataFile')
writeDataFile(trainInputImagesPath, trainOutputImagesPath, trainInputImagesFiles, trainOutputImagesFiles, trainDataFileName)
print(str(datetime.now()) + ': trainDataFile complete')

print(str(datetime.now()) + ': writing testDataFile')
writeDataFile(testInputImagesPath, testOutputImagesPath, testInputImagesFiles, testOutputImagesFiles, testDataFileName)
print(str(datetime.now()) + ': testDataFile complete')

print(str(datetime.now()) + ': writing validDataFile')
writeDataFile(validInputImagesPath, validOutputImagesPath, validInputImagesFiles, validOutputImagesFiles, validDataFileName)
print(str(datetime.now()) + ': validDataFile complete')

"""# Training and Testing"""

from datetime import datetime
import numpy as np
import tensorflow as tf

print(str(datetime.now()) + ': loading data files')
# Data sets
trainDataFileName = 'dataset/train.csv'
testDataFileName = 'dataset/test.csv'
validationDataFileName = 'dataset/valid.csv'
# Load datasets.
trainData = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=trainDataFileName,
    target_dtype=np.int,
    features_dtype=np.int)
testData = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=testDataFileName,
    target_dtype=np.int,
    features_dtype=np.int)
# validationData = tf.contrib.learn.datasets.base.load_csv_without_header(
#     filename=validationDataFileName,
#     target_dtype=np.int,
#     features_dtype=np.int)

trainingSteps = 1000
totalTrainingSteps = 5000

featureColumns = [tf.contrib.layers.real_valued_column("", dimension=75)]
hiddenUnits = [100, 150, 100, 50]
classes = 2
modelDir = 'model'
classifierConfig = tf.contrib.learn.RunConfig(save_checkpoints_secs = None, save_checkpoints_steps = trainingSteps)

classifier = tf.contrib.learn.DNNClassifier(feature_columns = featureColumns,
                                                hidden_units = hiddenUnits,
                                                n_classes = classes,
                                                model_dir = modelDir,
                                                config = classifierConfig)

# Define the training inputs
def getTrainData():
    x = tf.constant(trainData.data)
    y = tf.constant(trainData.target)
    return x, y

# Define the test inputs
def getTestData():
    x = tf.constant(testData.data)
    y = tf.constant(testData.target)
    return x, y

# Define the validation inputs
# def getValidationData():
#     x = tf.constant(validationData.data)
#     y = tf.constant(validationData.target)
#     return x, y

# print(str(datetime.now()) + ': training...')
# prevAccuracy = -1.0
# for i in range(totalTrainingSteps // trainingSteps):
#     classifier.fit(input_fn=getTrainData, steps=trainingSteps)
#     currentAccuracy = classifier.evaluate(input_fn=getValidationData, steps=1)['accuracy']
#     print(str(datetime.now()) + ': training: validation step: ' + str(i) + ' currentAccuracy:', currentAccuracy)
#     #if(currentAccuracy <= prevAccuracy):
#     #    break
#     prevAccuracy = currentAccuracy

print(str(datetime.now()) + ': training...')
classifier.fit(input_fn=getTrainData, steps=totalTrainingSteps)
print(str(datetime.now()) + ': testing...')
accuracy = classifier.evaluate(input_fn=getTestData, steps=1)['accuracy']
print(str(datetime.now()) + ': done')
print(str(datetime.now()) + ': accuracy:', accuracy)

"""## Accuracy = 81.7 %

# Test with seperate Input to Generate Output Mask
"""

from datetime import datetime
from PIL import Image
import sys
import numpy as np
import tensorflow as tf

if(len(sys.argv) != 3):
	raise Exception('invalid command line arguments')

print(str(datetime.now()) + ': initializing input data...')

rectSize = 5;

inputImagePath = 'dataset/TestingImage'
inputImageFile = '24478825_15.tiff'
inputImage = Image.open(inputImagePath + '/' + inputImageFile)
inputImageXSize, inputImageYSize = inputImage.size

outputImagePath = 'dataset/TestedOutput'
outputImageFile = 'Output'
outputImage = inputImage.crop((rectSize//2, rectSize//2, inputImageXSize - (rectSize//2), inputImageYSize - (rectSize//2)))
outputImageXSize, outputImageYSize = outputImage.size

print(str(datetime.now()) + ': initializing model...')
featureColumns = [tf.contrib.layers.real_valued_column("", dimension=75)]
# hiddenUnits = [100, 100, 100, 50]
# hiddenUnits = [100, 150, 200, 150, 100, 50]
hiddenUnits = [100, 150, 100, 50]
classes = 2
classifier = tf.contrib.learn.DNNClassifier(feature_columns = featureColumns,
												hidden_units = hiddenUnits,
												n_classes = classes,
												model_dir = 'model')

def extractFeatures():
    features = np.zeros((((inputImageXSize - ((rectSize//2)*2)) * (inputImageYSize - ((rectSize//2)*2))), rectSize*rectSize*3), dtype=np.int)
    rowIndex = 0
    
    for x in range(rectSize//2, inputImageXSize - (rectSize//2)):
        for y in range(rectSize//2, inputImageYSize - (rectSize//2)):            
            rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
            subImage = inputImage.crop(rect).load()
            colIndex = 0
            for i in range(rectSize):
                for j in range(rectSize):
                    features[rowIndex, colIndex] = subImage[i, j][0]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][1]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][2]
                    colIndex += 1
            
            rowIndex += 1
    
    return features
    
def constructOutputImage(predictions):
    outputImagePixels = outputImage.load()
    rowIndex = 0
    for x in range(outputImageXSize):
        for y in range(outputImageYSize):
            outputImagePixels[x, y] = ((255, 255, 255) if predictions[rowIndex] else (0, 0, 0))
            rowIndex += 1
        
print(str(datetime.now()) + ': processing image', inputImageFile)
predictions = list(classifier.predict(input_fn=extractFeatures))

print(str(datetime.now()) + ': constructing output image...')
constructOutputImage(predictions)

print(str(datetime.now()) + ': saving output image...')
outputImage.save(outputImagePath + '/' + outputImageFile, 'JPEG')

print(str(datetime.now()) + ': done')