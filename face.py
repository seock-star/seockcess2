import cv2
import argparse
import glob
import os


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, \
help = 'Path to the input image')
args = vars(ap.parse_args())
dataset= args["dataset"]
print('처리중...')
verbose=True
imagePath=os.path.join('D:\Emotion3\resultss')
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
for imagePath in glob.glob(dataset +"/*.jpg"):
 print(imagePath, '처리중...') 
 img = cv2.imread(imagePath)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.3,5)
 
 for (x,y,w,h) in faces:
    cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    # 이미지를 저장
    if cropped.size == 0: #얼굴감지 못했으면 생략하세요
     continue
    cropped = cv2.resize(cropped, dsize=(48,48), interpolation=cv2.INTER_AREA)

    
    
    loc1=imagePath.rfind("\\")
    loc2=imagePath.rfind(".")
    fname='results/'+imagePath[loc1+1:loc2]+'.png'
    cv2.imwrite(fname,cropped)
    cv2.destroyAllWindows()

# cv2.imshow('Image view', img)
# cv2.waitKey(0)
cv2.destroyAllWindows()
