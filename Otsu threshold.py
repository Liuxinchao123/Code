
# -*- coding: cp936 -*-
import os

import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

path = 'E:/yolov5-prune/shao/'
cut_path = 'E:/yolov5-prune/WSP-PNG/OTSU+B/S+OTSU+B/'

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
        retVal, a_img = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
        a_img = 255 - a_img
        print(a_img)
        kernel = np.ones((5, 5), np.uint8)
        # 闭运算
        closing = cv2.morphologyEx(a_img, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(temp + b + '.png', closing)

