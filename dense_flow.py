import numpy as np
import cv2
from util import *


def dense_flow(original_frame1, original_frame2, pyramid=0, min_dim=15, max_dim=60, thresh=150):
    frame1 = original_frame1.copy()
    frame2 = original_frame2.copy()

    bboxes = []
    if pyramid > 0:
        for _ in range(pyramid):
            frame1 = cv2.pyrDown(frame1)
            frame2 = cv2.pyrDown(frame2)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    
    opticalFlow = cv2.calcOpticalFlowFarneback(
        frame1, 
        frame2, 
        None, 
        pyr_scale=0.5, 
        levels=1, 
        winsize=20, 
        iterations=3, 
        poly_n=5, 
        poly_sigma=1.2, 
        flags=0
    )

    originalMag, originalAng = cv2.cartToPolar(opticalFlow[..., 0], opticalFlow[..., 1])

    originalMag = cv2.normalize(originalMag, None, 0, 255, cv2.NORM_MINMAX)
    # originalAng = originalAng * 180 / np.pi / 2
    # originalAng = cv2.normalize(originalAng, None, 0, 255, cv2.NORM_MINMAX)

    ret, magMask = cv2.threshold(originalMag, thresh, 255, cv2.THRESH_BINARY)
    # ret, angMask = cv2.threshold(originalAng, thresh, 255, cv2.THRESH_BINARY)

    if pyramid > 0:
        for _ in range(pyramid):
            magMask = cv2.pyrUp(magMask)
    
    magMask = np.uint8(magMask)


    # tmpAng = originalAng.copy()
    # speedAverage = np.average(tmpAng[originalMag > 100]) # average only moving pixels in the speed mask
    # tmpAng[np.logical_and(tmpAng < speedAverage + 45, tmpAng > speedAverage - 45)] = 0


    contours, hierarchy = cv2.findContours(magMask, 1, 2)

    output_frame = original_frame1.copy()
    # temp_mask = np.zeros_like(originalAng)


    # totalArea = 0
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     totalArea += area

    # if totalArea > 20000:
    #     contours, hierarchy = cv2.findContours(tmpAng, 1, 2)


    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
        
        # flow_angle = cv2.bitwise_and(originalAng, temp_mask)

        multiplier = pyramid + 1
        if( min_dim*multiplier < h < max_dim*multiplier and min_dim*multiplier < w < max_dim*multiplier ):
            center = np.array([[x + w / 2], [y + h / 2]])
            # centers.append(np.round(center))
            bbox = (x, y, w, h)
            bboxes.append(bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            # color = (255, 0, 255) if is_target else (255, 0, 0)

            output_frame = cv2.rectangle(output_frame, p1, p2, (255, 0, 0), 3)
            # cv2.putText(output_frame, f'{np.average(flow_angle)}', p1,  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)



    return output_frame, bboxes




