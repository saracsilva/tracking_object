import cv2
import numpy as np


cap = cv2.VideoCapture("Different_Bouncing_Ball_References.mp4")
_,frame = cap.read()

cv2.imshow("Frame", frame)
cv2.waitKey(0)