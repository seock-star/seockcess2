# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.png'):
 fileList = []
 print(myDir)
 for root, dirs, files in os.walk(myDir, topdown=False):
    for name in files:
        if name.endswith(format):
            fullName = os.path.join(root, name)
            fileList.append(fullName)
 return fileList



# load the original image
myFileList = createFileList('D:/Emotion3/result/')
num = 0;
train_img = np.float32(np.zeros((474, 48, 48, 3))) #사진갯수만큼
train_label = np.float64(np.zeros((474, 1)))
     
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()
    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    #f = open('output.csv', 'w', encoding='utf-8', newline='')
    with open("img_pixels.csv", 'a',newline='') as f:
     writer = csv.writer(f,delimiter=' ')   
     writer.writerow(value)
     # np.savetxt('np.csv', value, delimiter=',')