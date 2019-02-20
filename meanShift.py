'''
The Direction to the closest cluster centroid is
determined by where most of the points nearby are at.

MeanShift can be given a target to track, calculated the color histogram
of the target area, and then keep sliding the tracking window to the 
closest match(cluster center)
'''

import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frameCopy = frame.copy()
x1 = -1 
x2 = -1 
y1 = -1 
y2 = -1 
drawing = False 

def mouseCallBack(action,x,y,flags,userData):

    global x1,x2,y1,y2,drawing,frameCopy,frame
    if action == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        x1 = x
        y1 = y 

    elif action == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frameCopy = frame.copy()
            cv2.rectangle(frameCopy,(x1,y1),(x,y),(255,0,0),1)

    elif action == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2= x
        y2= y
        cv2.rectangle(frameCopy,(x1,y1),(x2,y2),(0,255,0),1) 

def selectRegion():
        
    global x1,x2,y1,y2,frameCopy,prev_frame
    cv2.namedWindow('Region')
    cv2.setMouseCallback('Region',mouseCallBack)

    while True:
        cv2.imshow('Region',frameCopy)
        k = cv2.waitKey(1) & 0xFF

        if k == 13 :
            break 
        
        elif k == ord('c'):
            x1 = -1 
            x2 = -1 
            y1 = -1
            y2 = -1
            frameCopy = frame.copy()
    
selectRegion()

track_window = (x1,y1,x2,y2)
roi = frame[y1:y2,x1:x2]

hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

terminationCriteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_EPS,10,1)

while True:

    ret, frame = cap.read()

    if ret == True:
        cv2.namedWindow('track_window')
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.meanShift(dst,track_window,terminationCriteria)

        x,y,w,h = track_window

        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)

        cv2.imshow('img',img2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()