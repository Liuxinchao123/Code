import os

import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

path = 'E:/yolov5-prune/123123123134124124124/'
cut_path = 'E:/yolov5-prune/131241414141414/'

for (root, dirs, files) in os.walk(path):
    temp = root.replace(path, cut_path)
    if not os.path.exists(temp):
        os.makedirs(temp)
    for file in files:
        a = file.split(".", 1)
        b = a[0]
        img = cv2.imread(os.path.join(root, file))
        # _, threshold_image = cv2.threshold(a_channel_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret1, thresh1 = cv2.threshold(gray, 198, 255, cv2.THRESH_BINARY_INV)
        thresh1 = 255 - thresh1
        kernel = np.ones((1, 1), np.uint8)
        # 闭运算
        closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        print(temp)
        print(b)
        print(ret1)
        print(thresh1)
        #cv2.imshow("1", binary)
        #cv2.waitKey(0)
        cv2.imwrite(temp + b + '.png', closing)
