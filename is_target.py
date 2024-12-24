import numpy as np
import cv2
from k_means import kmeans
from dense_flow import dense_flow


def get_mask(image):
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


def get_target_center(image):
    contours, hierarchy = cv2.findContours(image, 1, 2)
    width, height = image.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or x+w == width or y+h == height:
            print("not target")
            continue

        new_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.imshow("new", new_image)
        cv2.waitKey(0)
        return (x+w/2,y+h/2)


frame1 = cv2.imread("../data/frame12.png")
frame2 = cv2.imread("../data/frame13.png")

# cv2.imshow("frame1",frame1)
# cv2.imshow("frame2", frame2)

flow, bboxes = dense_flow(frame1, frame2)

cv2.imshow("flow", flow)
print(bboxes)


mask2 = get_mask(frame2)

gray_mask = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

new_center = get_target_center(gray_mask)
print(new_center)

bgdmodel = np.zeros((1, 65), np.float64)
fgdmodel = np.zeros((1, 65), np.float64)

cv2.grabCut(frame2, None, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)

cv2.imwrite("mask2.png", get_mask(frame2))

# cv2.waitKey(0)

cv2.destroyAllWindows()





