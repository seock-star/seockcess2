from keras.preprocessing.image import img_to_array 
import imutils
import cv2
from keras.models import load_model
import numpy as np
from PIL import ImageFont, ImageDraw, Image


face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
emotion_model_path = 'models/_mini_XCEPTION.31-0.67.hdf5' #모델 경로
emotion_classifier = load_model(emotion_model_path, compile=False)
predicted_class=10
me="설정"
img = cv2.imread('mom7.jpg')#이름 수정할것
# cv2.imshow('Image view', img)
# cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
face_array=[]
emotion=[]
for (x,y,w,h) in faces:
    if faces.size == 0: #얼굴감지 못했으면 생략하세요
     continue
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype("float") / 255.0
    roi_gray = img_to_array(roi_gray)
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_color = img[y:y+h, x:x+w]
    face_array.append(roi_gray)
  
    

for i in range(len(face_array)):  
 prediction = emotion_classifier.predict(face_array[i])
 predicted_class = np.argmax(prediction[0]) # 예측된 클래스 0, 1, 2
 print(prediction[0])
 print(predicted_class)
 if predicted_class == 0:
        me = "angry"
 elif predicted_class == 1:
        me = "disgust"
 elif predicted_class == 2:
        me = "fear"
 elif predicted_class == 3:
        me = "happy"
 elif predicted_class == 4:
        me = "sad"
 elif predicted_class == 5:
        me = "surprised"
 elif predicted_class == 6:
        me = "netural"
 elif predicted_class == 7:
        me = "drowsy"
 emotion.append(me) 



    # cropped = img_to_array(cropped)
    # cropped = np.expand_dims(cropped, axis=0)
    # cropped = preprocess_input(cropped)
    


for i in range(len(face_array)):         
 point =150*i,30
 cv2.putText(img,emotion[i],point,cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA) #알아서 수정하세용 ㅎ[이미지,글자,폰트,크기,색깔,두께,라인타입이라네요]   
 
    
 # press "Q" to stop
cv2.imshow('Image view', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
