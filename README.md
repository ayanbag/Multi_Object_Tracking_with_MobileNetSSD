# Multi Object Tracking Using MobileNet SSD 

Implementation of Multi Object Tracking Using pretrained MobileNet SSD using dlib library and OpenCV in Python. Here in this code we are using MobileNet SSD as the feature extractor.

## Output

![](gif/traffic.gif)


## Requirements :

- dlib
- opencv-python
- imutils

## Usage :

- Clone this Repository
```
git clone https://github.com/ayanbag/Multi_Object_Tracking_with_MobileNetSSD.git
cd Multi_Object_Tracking_with_MobileNetSSD
```
Then run the following command to install the required dependencies.
```
pip install -r requirements.txt
```

- Now excute the following command :

```
python multi_object_tracking.py -i <path-to-input>
```


**Note:** Our script processes the following command line arguments at runtime:

- `--input` or `-i` : The path to the input video file. We’ll perform multi-object tracking with dlib on this video.
- `--confidence` or `-c` : An optional override for the object detection confidence threshold of 0.2 . This value represents the minimum probability to filter weak detections from the object detector.
