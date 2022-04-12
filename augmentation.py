from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import argparse
import numpy as np
import os
import sys
import glob

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, \
help = 'Path to the input image')
args = vars(ap.parse_args())
dataset= args["dataset"]
print('처리중...')
verbose=True
# img = load_img('fear.jpg')  # PIL 이미지


# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# 지정된 `preview/` 폴더에 저장합니다.

imagePath=os.path.join('D:\Emotion3\results')

for imagePath in glob.glob(dataset +"/*.png"): 
  if cv2.waitKey(1) & 0xFF ==ord("q"):
           break        
  print(imagePath, '처리중...')
  i = 0
  src= cv2.imread(imagePath)
  src = img_to_array(src)  # (3, 48, 48) 크기의 NumPy 배열
  src = src.reshape((1,) + src.shape)  # (1, 3, 48, 48) 크기의 NumPy 배열
  for imagePath in datagen.flow(src, batch_size=1,
                          save_to_dir='results', save_prefix='cat', save_format='png'):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성하고 마칩니다
