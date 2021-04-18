import re
import shutil
import requests

trainInputImagesPath = 'dataset/Training/Input/'
trainOutputImagesPath = 'dataset/Training/InputLables'
testInputImagesPath = 'dataset/Testing/testingInput'
testOutputImagesPath = 'dataset/Testing/testingOutput'
validInputImagesPath = 'dataset/Validation/validationInput'
validOutputImagesPath = 'dataset/Validation/validationOutput'


qbfile = open("data.txt", "r")

counter = 0;
currentPath = "E:\\Datasets\\SateliteToMaps\\" + trainInputImagesPath;
setName = "trainSat"
for aline in qbfile:
    imagePath = re.search('href="(.*)">', aline).group(1)
    filenpath =currentPath + setName + str(counter) + ".tiff";
    counter += 1;
    response = requests.get(imagePath)


qbfile.close()