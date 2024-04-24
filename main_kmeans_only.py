import numpy as np
import cv2
from dense_flow import dense_flow
from k_means import *
from tracker2 import Tracker
from util import *
import math

# use kmeans all the way


file_name="DRONE_113"
cap = cv2.VideoCapture(f"../Data/Video_V/V_{file_name}.mp4")
# cap = cv2.VideoCapture(f"../Data/Video_IR/IR_{file_name}.mp4")


ret, frame1 = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), int(fps), (int(width), int(height)) )

# check if the file exists
if not ret:
    print("invalid file")

counter = 0

tracker = Tracker(
    dist_thresh=50, 
    max_frames_to_skip=5, 
    max_trace_length=10, 
    trackIdCount=0
)

while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    timer = cv2.getTickCount()

    new_target_img, new_target_boxes = kmeans_get_targets(frame2, pyramid=False)
    tracker.update(new_target_boxes, frame2)

    img = draw_tracks(frame2, tracker)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    cv2.imshow('Original', img)
    cv2.imshow("mask", new_target_img)

    writer.write(img)

    key = cv2.waitKey(1)
    if key == 27:
        break  # ESC key pressed

    frame1 = frame2

    counter += 1



writer.release()
cap.release()
cv2.destroyAllWindows()

