import numpy as np
import cv2
import json

img_name = 'thumb_cropped_id_front (21)_censored'
img_dir = './uat/'
img_type = '.jpg'

# read detection settings from json
file = open('./settings/' + img_name + '.json')
settings = json.load(file)

# assign id detection settings
threshold_min = settings['threshold_min']
threshold_max = settings['threshold_max']
contour_area_min = settings['contour_area_min']
img_src = img_dir + img_name + img_type

img_color = cv2.imread(img_src, cv2.IMREAD_COLOR)
img_gray = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)

_,threshold = cv2.threshold(img_gray, threshold_min, threshold_max, cv2.THRESH_BINARY)
contours,_ = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
boxes = { }
validBox = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if (area > contour_area_min):
        epsilon = 0.025 * cv2.arcLength(cnt, True);
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if (len(approx) == 4):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            validBox += 1
            boxes['box' + str(validBox)] = box.tolist()
            cv2.drawContours(img_color, [box], 0, (255, 0, 0), 2)

cv2.imshow('image', img_color)

with open('./output/' + img_name + '.json', 'w') as outfile:
    json.dump(boxes, outfile)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
