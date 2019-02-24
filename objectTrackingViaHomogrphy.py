import numpy as np
import cv2 

cap = cv2.VideoCapture(0)
#object = selectObject()
sift = cv2.xfeatures2d.SIFT_create()

objectKeypoints = sift.detectAndCompute(object,None)
while True:

    ret, frame = cap.read()
    gFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Tracking',frame)

    k = cv2.waitKey(1)
    if k==27:
        break


cv2.destroyAllWindows()
cap.release()