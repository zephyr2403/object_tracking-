# Object Tracking 

## Approach Used:
* **Mean Shift** : Meanshift can be given a target to track, it calculates the color histogram of the target area, and then keep sliding the tracking window to the closest match *(cluster center)* .
* **CAMShift** : Just using mean shift won't change the window size if the target moves away or towards the camera, use CAMshift to update the size of window. 
* **Via Homography**: Find the keypoints from the target and main windom, after keypoints matching utilise homography to find the target window in the main window.

## Requirements:
* Python 2.7
* opencv
* Object Tracking with homography requires *SIFT* which is in **opencv-contrib**.
## Usage 
* Run via `python filename.py`
* Select a rectangular window *(with help of mouse)* which you want to track and hit `Enter`.

## Result
#### Tracking notebook.
### MeanShift
<p align="center">
<img src = 'outputs/meanShift.gif' />
</p>

### CamShift
<p align="center">
<img src = 'outputs/camShift.gif' />
</p>


### Via Homography
<p align="center">
<img src = 'outputs/viahomo.gif' />
</p>
