import cv2
import numpy as np


cap = cv2.VideoCapture("Different_Bouncing_Ball_References.mp4")

# Object Detection from a stable camera - removing background
object_detector = cv2.createBackgroundSubtractorMOG2()



while True:
    _, frame = cap.read()

    #Object Detection

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    contours, _ =  cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        #Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            x,y,w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3 )

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()