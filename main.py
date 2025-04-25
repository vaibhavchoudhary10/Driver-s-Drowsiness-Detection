import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

model_path = r'D:\Python Projects\Drowsiness_Detection\Models\model.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#model = load_model(r'D:\Python Projects\Drowsiness_Detection\Models\model.h5')

mixer.init()
sound = mixer.Sound(r'D:\Python Projects\Drowsiness_Detection\alarm.wav')
cap = cv2.VideoCapture(0)
Score = 0
while True:
    ret, frame = cap.read()
    height, width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    cv2.rectangle(frame, (0,height-50), (200,height),color=(0,0,0), thickness=cv2.FILLED)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=3)

    for(ex,ey,ew,eh) in eyes:
        #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
        
        # preprocessing steps
        eye = frame[ey:ey+eh, ex:ex+ew]
        eye = cv2.resize(eye, (80, 80))
        eye  = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye, axis=0)

        # prediction
        prediction = model.predict(eye)

        #if eyes are closed
        if prediction[0][0] > 0.30:
            cv2.putText(frame, "Closed",(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            
            Score = Score + 1
            if Score > 15:
                try:
                    sound.play()
                except:
                    pass

        elif prediction[0][1] > 0.90:
            cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)      
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            
            Score = Score - 1
            if Score < 0:
                Score = 0        


    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    