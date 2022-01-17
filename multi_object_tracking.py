from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PyInquirer import style_from_dict, Token, prompt
import time
import os

if os.path.isdir("output") is False: os.mkdir("output")

# classes to track
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

# initialize the list of class labels MobileNet SSD was trained to
style = style_from_dict({ Token.QuestionMark: '#E91E63 bold', Token.Selected: '#00FFFF', Token.Instruction: '', Token.Answer: '#2196f3 bold', Token.Question: '#7FFF00 bold',})
time.sleep(0.2)
class_option=[ 
	{
		'type':'list',
		'name':'class',
		'message':'Class for tracking:',
		'choices': CLASSES,
	}
]
class_answer=prompt(class_option,style=style)
class_to_track=class_answer['class']

proto = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video file")
#ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto, model)
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"])
writer = None

trackers = []
labels = []

if "/" in list(args["input"]):
    output = args["input"].split(".")[0].split("/")[-1]
else:
    output = args["input"].split(".")[0]

fps = FPS().start()

while True:
	(grabbed, frame) = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/"+output+".avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	if len(trackers) == 0:
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]
				if CLASSES[idx] != class_to_track:
					continue
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				t.start_track(rgb, rect)
				labels.append(label)
				trackers.append(t)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	else:
		for (t, l) in zip(trackers, labels):
			t.update(rgb)
			pos = t.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	if writer is not None:
		writer.write(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()
cv2.destroyAllWindows()
vs.release()