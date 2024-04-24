import numpy as np
import cv2
from dense_flow import *
from k_means import kmeans
from tracker2 import Tracker
from util import *
import math
from classifier.model import get_model
from tensorflow import keras
from tensorflow.keras.models import load_model
import time


# confirm initial points using cnn


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

model = load_model("classifier_single.keras")

tracker = Tracker(
    dist_thresh=50, 
    max_frames_to_skip=5, 
    max_trace_length=10, 
    trackIdCount=0,
)



def is_target_cnn(frame2, box, model):
    x, y, w, h = box

    crop1 = cv2.resize(imcrop(frame1, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)

    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

    crop1 = cv2.filter2D(crop1,-1,np.array(kernel))

    prediction = model.predict(np.array([crop1]))
    prediction = prediction[0] == max(prediction[0])
    target_class = "target" if prediction[0] else "noise"
    print(target_class)
    return target_class == "target", box


target_count = 0

while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    # kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    # frame2 = cv2.filter2D(frame2,-1,np.array(kernel))
        
    start = time.time()

    # get dense optical flow boxes
    dense_image, bboxes = dense_flow(frame1, frame2, pyramid=0)
    
    end = time.time()
    print(f"elapsed time {end - start}")

    target_boxes = []

    for index, bbox in enumerate(bboxes):
        x, y, w, h = bbox

        is_target, box2 = is_target_cnn(frame2, bbox, model)
        if is_target:
            target_boxes.append(box2)

            target_count += 1

            # p1 = (int(bbox[0]), int(bbox[1]))
            # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # dense_image = cv2.rectangle(dense_image, p1, p2, (0, 0, 255), 3)




    tracker.update(target_boxes, frame1)

    # draw tracker prodictions
    img = draw_tracks(frame1, tracker)


    cv2.imshow('Original', img)

    # cv2.imwrite(f"output_images/{file_name}_{counter}.png", img)


    # writer.write(img)

    key = cv2.waitKey(1)
    if key == 27:
        break  # ESC key pressed



    frame1 = frame2
    counter += 1

print(f"target count: {target_count}")

writer.release()
cap.release()
cv2.destroyAllWindows()

