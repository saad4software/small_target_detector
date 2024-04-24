import numpy as np
import cv2
from util import *


def kmeans(img, k):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    return res2


def kmeans_get_targets(image, pyramid = False):
    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    processed = cv2.filter2D(image, -1, np.array(kernel))

    if pyramid : processed = cv2.pyrDown(processed)
    mask = kmeans_get_mask(processed)
    if pyramid : mask = cv2.pyrUp(mask)
    new_img, bboxes = get_contour_bboxes(mask)
    # new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel1, iterations=1)

    # new_img = draw_boxes(image, bboxes)
    # cv2.imshow("new_img", mask)

    final_results = []
    for box in bboxes:
        x, y, w, h = box


        if  w > 20 or h > 20 :
            continue

        # expanded_box = [int(x-4), int(y-4), int(w+8), int(h+8)]
        is_target, new_box = kmeans_is_target(image, box)
        # is_target, new_box = kmeans_is_target_mask(mask, box)

        if is_target:
            print(box)
            print("is target")

            final_results += [new_box]
    
    # test = draw_boxes(image, final_results)
    # cv2.imshow("test", test)
    return new_img, final_results


def kmeans_get_mask(image):
    img_frame = kmeans(image.copy(), 2)
    image_colors = np.unique(img_frame)

    count1 = len(img_frame[img_frame==image_colors[0]])
    count2 = len(img_frame[img_frame==image_colors[1]])

    target_color = image_colors[0] if count1 < count2 else image_colors[1]

    mask = cv2.inRange(img_frame, np.array(target_color), np.array(target_color))

    img_frame = cv2.cvtColor(img_frame,cv2.COLOR_GRAY2RGB)
    img_frame = cv2.bitwise_and(img_frame, img_frame, mask=mask)
    ret, msk_frame = cv2.threshold(img_frame, 20, 255, cv2.THRESH_BINARY)
    return msk_frame


def kmeans_get_target_mask(image):
    img_new = kmeans(image.copy(), 2)

    border1 = img_new[:2, :]
    border2 = img_new[-2:, :]
    border3 = img_new[:, :2]
    border4 = img_new[:, -2:]

    count1 = len(np.unique(border1))
    count2 = len(np.unique(border2))
    count3 = len(np.unique(border3))
    count4 = len(np.unique(border4))
            
    border_color = np.unique(border1)
    image_colors = np.unique(img_new)

    total = count1 + count2 + count3 + count4
    is_target = total == 4 and len(image_colors) > 1

    target_color = image_colors[1] if image_colors[0] == border_color else image_colors[0]
    mask = cv2.inRange(img_new, np.array(target_color), np.array(target_color))

    return is_target, mask


def kmeans_is_target(image, box):
    x, y, w, h = box
    # cv2.imwrite(f"img.png", image)


    crop = imcrop(image, (x, y, x + w, y + h))


    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

    mask_image = np.zeros_like(image)

    if crop.any():
        crop = cv2.filter2D(crop,-1,np.array(kernel))
        
        img_new = kmeans(crop, 2)
        # cv2.imwrite(f"crop{box}.png", img_new)


        # cv2.imwrite(f"img{box}.png", img_new)

        border1 = img_new[:2, :]
        border2 = img_new[-2:, :]
        border3 = img_new[:, :2]
        border4 = img_new[:, -2:]

        count1 = len(np.unique(border1))
        count2 = len(np.unique(border2))
        count3 = len(np.unique(border3))
        count4 = len(np.unique(border4))

        color = (0, 0, 255)
            
        border_color = np.unique(border1)
        image_colors = np.unique(img_new)

        total = count1 + count2 + count3 + count4
        is_target = total == 4 and len(image_colors) > 1

        new_box = box

        if is_target and img_new.shape == (w, h):
            target_color = image_colors[1] if image_colors[0] == border_color else image_colors[0]
            mask = cv2.inRange(img_new, np.array(target_color), np.array(target_color))

            img_new = cv2.cvtColor(img_new,cv2.COLOR_GRAY2RGB)
            img_new = cv2.bitwise_and(img_new,img_new, mask=mask)
            mask_image[y:y+h, x:x+w] = img_new
            ret, mask_image = cv2.threshold(mask_image, 20, 255, cv2.THRESH_BINARY)
            gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            new_center = get_contour_center(gray_mask)
            new_box = (int(new_center[0]-w/2), int(new_center[1]-h/2), w, h) if new_center else box


        return is_target, new_box


    return False, box


def kmeans_is_target_image(image):
    if not image:
        return False
        
    img_new = kmeans(crop, 2)

    border1 = img_new[:2, :]
    border2 = img_new[-2:, :]
    border3 = img_new[:, :2]
    border4 = img_new[:, -2:]

    count1 = len(np.unique(border1))
    count2 = len(np.unique(border2))
    count3 = len(np.unique(border3))
    count4 = len(np.unique(border4))
            
    border_color = np.unique(border1)
    image_colors = np.unique(img_new)

    total = count1 + count2 + count3 + count4
    is_target = total == 4 and len(image_colors) > 1

    new_box = box

    if is_target and img_new.shape == (w, h):
        target_color = image_colors[1] if image_colors[0] == border_color else image_colors[0]
        mask = cv2.inRange(img_new, np.array(target_color), np.array(target_color))

        img_new = cv2.cvtColor(img_new,cv2.COLOR_GRAY2RGB)
        img_new = cv2.bitwise_and(img_new,img_new, mask=mask)
        mask_image[y:y+h, x:x+w] = img_new
        ret, mask_image = cv2.threshold(mask_image, 20, 255, cv2.THRESH_BINARY)
        gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        new_center = get_contour_center(gray_mask)
        new_box = (int(new_center[0]-w/2), int(new_center[1]-h/2), w, h) if new_center else box


    return is_target, new_box




def kmeans_is_target_mask(mask, box):
    x, y, w, h = box

    e1 = int(20/ w)
    e2 = int(20/ h)

    crop = imcrop(mask, (x - int(e1/2), y - int(e2/2), x + 20, y + 20))

    if crop.any():

        border1 = crop[:2, :]
        border2 = crop[-2:, :]
        border3 = crop[:, :2]
        border4 = crop[:, -2:]

        count1 = len(np.unique(border1))
        count2 = len(np.unique(border2))
        count3 = len(np.unique(border3))
        count4 = len(np.unique(border4))
            
        border_color = np.unique(border1)
        image_colors = np.unique(crop)

        total = count1 + count2 + count3 + count4
        is_target = total == 4 and len(image_colors) > 1

        new_box = box

        return is_target, new_box


    return False, box


def is_target_kmeans(frame1, frame2, box, model):
    x, y, w, h = box

    crop1 = cv2.resize(imcrop(frame1, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)
    crop2 = cv2.resize(imcrop(frame2, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)

    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

    crop1 = cv2.filter2D(crop1,-1,np.array(kernel)) 
    crop2 = cv2.filter2D(crop2,-1,np.array(kernel)) 

    mask1 = kmeans_get_target_mask(crop1)
    mask2 = kmeans_get_target_mask(crop2)


    return target_class == "target", box


