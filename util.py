import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize

def is_inside(point, rect):
    return rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]


def boxes2centers(bboxes):
    centers = []
    for box_index, box in enumerate(bboxes):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        center = [[center_x], [center_y]]
        centers += [center]

    return centers

def centers_distance(center1, center2):
    diff = np.array([center1]) - np.array(center2)
    distance = np.sqrt(diff[0][0]*diff[0][0] + diff[0][1]*diff[0][1])
    return distance


def is_inside(point, rect):
    return rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]



def draw_tracks_centroid(frame, tracker):
    for track in tracker.tracks:

        if not track.trace: break

        trace_x = track.trace[-1][0][0]
        trace_y = track.trace[-1][1][0]

        cv2.putText(frame, f'ID: {track.track_id}', (int(trace_x), int(trace_y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


def draw_tracks(frame, tracker):
    output = frame.copy()
    for track in tracker.tracks:
        if not len(track.trace)>=5: break

        for b in track.trace:
            pt1 = (int(b[0] + b[2]/2), int(b[1]+b[3]/2))
            cv2.circle(output, pt1, 0, [255, 255, 0], 2)

        center = box2center(track.trace[-1])
        bbox = track.trace[-1]
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        box_color = (255, 255, 0) if track.is_target() else (255, 255, 255)

        output = cv2.rectangle(output, p1, p2, box_color, 1)
        cv2.putText(output, f'ID: {track.track_id}', (int(center[0]), int(center[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    return output


def draw_points(frame, points):
    for pt in points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 0, [255, 0, 0], 2)
    return frame


def draw_boxes(frame, bboxes):
    for box in bboxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return frame


def get_contour_center(image):

    contours, hierarchy = cv2.findContours(image, 1, 2)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        new_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        return (x+w/2,y+h/2)


def get_contour_bboxes(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = cv2.pyrDown(image)
    # image = cv2.pyrUp(image)
    # ret, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)


    contours, hierarchy = cv2.findContours(image, 1, 2)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes += [[x - 4, y - 4, w + 8, h + 8]]
        new_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    return new_image, bboxes


def get_target_center(image, box):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image, 1, 2)
    box_x, box_y, box_w, box_h = box

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x == box_x or y == box_y or x+w == box_w or y+h == box_h:
            print("not target")
            continue

        # new_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # cv2.imshow("new", new_image)
        # cv2.waitKey(0)
        return (x+w/2,y+h/2)


def is_target_cnn_pairs(frame1, frame2, box, model):
    x, y, w, h = box

    crop1 = cv2.resize(imcrop(frame1, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)
    crop2 = cv2.resize(imcrop(frame2, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)

    # cv2.imwrite(f"crop{box}1.png", crop1)
    # cv2.imwrite(f"crop{box}2.png", crop2)

    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    # frame2 = cv2.filter2D(frame2,-1,np.array(kernel))
    # cv2.imwrite(f"crop{box}3.png", cv2.filter2D(crop1,-1,np.array(kernel)))
    # cv2.imwrite(f"crop{box}4.png", cv2.filter2D(crop2,-1,np.array(kernel)))

    crop1 = cv2.filter2D(crop1,-1,np.array(kernel))
    crop2 = cv2.filter2D(crop2,-1,np.array(kernel))

    # _, mask1 = kmeans_get_target_mask(crop1)
    # _, mask2 = kmeans_get_target_mask(crop2)

    # cv2.imwrite(f"crop{box}5.png", cv2.filter2D(mask1,-1,np.array(kernel)))
    # cv2.imwrite(f"crop{box}6.png", cv2.filter2D(mask2,-1,np.array(kernel)))

    prediction = model.predict(np.array([[crop1, crop2]]))
    prediction = prediction[0] == max(prediction[0])
    target_class = "target" if prediction[0] else "noise"
    print(target_class)
    return target_class == "target", box


def boxes2centers2(bboxes):
    centers = []
    for box_index, box in enumerate(bboxes):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        center = [center_x, center_y]
        centers += [[center]]

    return centers


def box2center(box):
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    center = [center_x, center_y]
    return center


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def imcrop_test(img, bbox):
    x1, y1, x2, y2 = bbox
    is_target = True
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        is_target = False
    return img[y1:y2, x1:x2, :], is_target

def imcrop2d(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]


