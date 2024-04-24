import numpy as np
import cv2
from dense_flow import *
from k_means import kmeans_is_target
from tracker2 import Tracker
from util import *
import math
from classifier.model import get_model
from tensorflow import keras
from tensorflow.keras.models import load_model

import time

# confirm initial points using kmeans

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


model = load_model("classifier_single.keras")


while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    # kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    # frame2 = cv2.filter2D(frame2,-1,np.array(kernel))
        
    
    # get dense optical flow boxes
    dense_image, bboxes = dense_flow(frame1, frame2, pyramid=0)
    

    target_boxes = []

    start = time.time()

    for index, box in enumerate(bboxes):
        x, y, w, h = box

        is_target, box2 = kmeans_is_target(frame2, box)

        if is_target:
            target_boxes.append(box2)


    tracker.update(target_boxes, frame1)
        

    # draw tracker prodictions
    img = draw_tracks(frame1, tracker)
    

    cv2.imshow('Original', img)

    # cv2.imwrite(f"output_images/{file_name}_{counter}.png", img)


    writer.write(img)

    key = cv2.waitKey(1)
    if key == 27:
        break  # ESC key pressed

    # end = time.time()
    # print(f"elapsed time {end - start}")

    frame1 = frame2
    counter += 1



writer.release()
cap.release()
cv2.destroyAllWindows()


