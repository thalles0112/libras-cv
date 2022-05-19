import cv2 as cv
from keras.models import load_model
import mediapipe as mp
import numpy as np
import os

mpDraw = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)
size = (300,300)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
model = load_model('modellines')
labels = os.listdir()
frame = []
while True:
    success, img = cap.read()
    try:
        img = cv.resize(img, size)
    except:
        break
    
    if success:
        
        results =  hands.process(img)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                for id, lm in enumerate(handLms.landmark):
                    
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmsid = [id, cx, cy]        
                    frame.append(lmsid)
                
                frame = np.array(frame)
                frame = np.expand_dims(frame, 0)
                print(frame.shape)
                
                pred = np.argmax(model.predict(frame))
                print(pred)

                frame = []
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  
        
    cv.imshow('a', img)
    cv.waitKey(1)

