import face_recognition
import cv2
import numpy as np
import time
import sys
# from darkflow.net.build import TFNet
import darkflow_repo.run as func

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    stime = time.time()
    ret, unt_frame = capture.read()
    frame = unt_frame
    print("new frame")
    if ret:
        finalFrame = func.yolo(frame, unt_frame)
        cv2.imshow('frame', finalFrame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()