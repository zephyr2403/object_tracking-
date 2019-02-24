import numpy as np
import cv2 

'''VARIABLES TO SELECT OBJECT <RETANGULAR REGION>'''
x1 = -1 
x2 = -1 
y1 = -1 
y2 = -1 
drawing = False 

'''MOUSE CALLBACK TO SELECT FOUR CORNER OF THE OBJECT'''
def mouseCallBack(action,x,y,flags,userData):

    global x1,x2,y1,y2,drawing,firstFrame,frameCopy
    if action == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        x1 = x
        y1 = y 

    elif action == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frameCopy = firstFrame.copy()
            cv2.rectangle(frameCopy,(x1,y1),(x,y),(255,0,0),1)

    elif action == cv2.EVENT_LBUTTONUP:
        drawing = False
        frameCopy = firstFrame.copy()
        x2= x
        y2= y
        cv2.rectangle(frameCopy,(x1,y1),(x2,y2),(0,255,0),1) 

'''
FUNCTION TO ALLOW USER TO SELECT OBJECT FROM THE FRAME
'''

def selectObject():
        
    global x1,x2,y1,y2,frameCopy,firstFrame
    frameCopy = firstFrame.copy()
    cv2.namedWindow('Select Object by drag mouse. Press c to clear,Enter to Continue')
    cv2.setMouseCallback('Select Object by drag mouse. Press c to clear,Enter to Continue',mouseCallBack)

    while True:
        cv2.imshow('Select Object by drag mouse. Press c to clear,Enter to Continue',frameCopy)
        k = cv2.waitKey(1) & 0xFF

        if k == 13 :
            break 
        
        elif k == ord('c'):
            x1 = -1 
            x2 = -1 
            y1 = -1
            y2 = -1
            frameCopy = firstFrame.copy()
    
    obj = firstFrame[y1:y2,x1:x2]
    gObj = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
    return obj,gObj

#capture frames from webcamera
cap = cv2.VideoCapture(0)

ret, firstFrame = cap.read()
frameCopy = firstFrame.copy()

#Selecting Object to be tracked 
object,gObject = selectObject()

sift = cv2.xfeatures2d.SIFT_create()

objectKeypts, objectDesc = sift.detectAndCompute(gObject,None)#None=Mask

indexParams = dict(algorithm=0,trees=5)
searchParams = dict()
    
flann = cv2.FlannBasedMatcher(
    indexParams,
    searchParams
)

while True:

    ret, frame = cap.read()
    gFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gFrameKeypts,gFrameDesc = sift.detectAndCompute(gFrame,None)

    ###########################
    ### Visualize KeyPoints ###
    ###########################
    """   
    viewFrameKeypts = cv2.drawKeypoints(frame,gFrameKeypts,frame)
    viewobjectKeypts = cv2.drawKeypoints(object,objectKeypts,object)
    cv2.imshow('View Frame Key Points',viewFrameKeypts)
    cv2.imshow('View Object Key Points',viewobjectKeypts)
    """

    ''' FEATURE MATCHING '''

    matches = flann.knnMatch(objectDesc,gFrameDesc,k=2)

    goodMatches = []

    for a,b in matches:
        if a.distance < b.distance * .8:
            goodMatches.append(a)

    matchImg = cv2.drawMatches(
    gObject,
    objectKeypts,
    gFrame,
    gFrameKeypts,
    goodMatches,
    gFrame
    )    

    #Uncomment to see the matched features
    #cv2.imshow('Matches',matchImg)
    
    ''' HOMOGRAPHY '''

    if len(goodMatches) > 10 :
        object_pts = np.float32([objectKeypts[p.queryIdx].pt for p in goodMatches]).reshape(-1,1,2)
        frame_pts = np.float32([gFrameKeypts[p.trainIdx].pt for p in goodMatches]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(object_pts,frame_pts,cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = gObject.shape
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

        prep = cv2.perspectiveTransform(pts,matrix)

        tracking = cv2.polylines(frame,[np.int32(prep)],True,(123,123,123),3)
        cv2.imshow('Tracking',tracking)
    else:
        cv2.imshow('Tracking',frame)


    k = cv2.waitKey(1)
    #Break when Esc is pressed
    if k==27:
        break


cv2.destroyAllWindows()
cap.release()