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
classifierConfig = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=trainingSteps)

classifier = tf.contrib.learn.DNNClassifier(feature_columns=featureColumns,
                                            hidden_units=hiddenUnits,
                                            n_classes=classes,
                                            model_dir=modelDir,
                                            config=classifierConfig)


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

from datetime import datetime
from PIL import Image
import sys
import numpy as np
import tensorflow as tf

def extractFeatures():
    features = np.zeros((((inputImageXSize - ((rectSize//2)*2)) * (inputImageYSize - ((rectSize//2)*2))), rectSize*rectSize*3), dtype=np.int)
    rowIndex = 0

    for x in range(rectSize // 2, inputImageXSize - (rectSize // 2)):
        for y in range(rectSize // 2, inputImageYSize - (rectSize // 2)):
            rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
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


rectSize = 5;

inputImagePath = 'Predictions/input'
outputImagePath = 'Predictions/output'

inputImageFile = '22978990_15.tiff'
inputImage = Image.open(inputImagePath + '/' + inputImageFile)
inputImageXSize, inputImageYSize = inputImage.size

outputImageFile = 'Output' + inputImageFile + '.JPG'
outputImage = inputImage.crop((rectSize//2, rectSize//2, inputImageXSize - (rectSize//2), inputImageYSize - (rectSize//2)))
outputImageXSize, outputImageYSize = outputImage.size

print(str(datetime.now()) + ': processing image', inputImageFile)
predictions = list(classifier.predict(input_fn=extractFeatures))

print(str(datetime.now()) + ': constructing output image...')
constructOutputImage(predictions)

print(str(datetime.now()) + ': saving output image...')
outputImage.save(outputImagePath + '/' + outputImageFile, 'JPEG')

print(str(datetime.now()) + ': done')

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


print(str(datetime.now()) + ': processing image', inputImageFile)
predictions = list(classifier.predict(input_fn=extractFeatures))

print(str(datetime.now()) + ': constructing output image...')
constructOutputImage(predictions)

print(str(datetime.now()) + ': saving output image...')
outputImage.save(outputImagePath + '/' + outputImageFile, 'JPEG')

print(str(datetime.now()) + ': done')


"""## Accuracy = 81.7 %

# Test with seperate Input to Generate Output Mask
"""
