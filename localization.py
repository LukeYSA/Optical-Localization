import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple

from EdgeDetection import EdgeDetection
from ObjectDetection import ObjectDetection

if __name__ == '__main__':
    path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
    output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

    img = cv2.imread(path + '/chess_irl.png')
    # Initialize edge detection and pixel coordinate to real coordinate mapping
    detect = EdgeDetection(img, output_dir=output_dir)

    # Initialize object detection process
    object = ObjectDetection(img, output_dir=output_dir)
    cx, cy = object.findObjectCenter()
    print('cx: ' + str(cx))
    print('cy: ' + str(cy))
    copy = img.copy()
    copy = cv2.circle(copy, (cx, cy), radius=7, color=(255, 0, 0), thickness=-1)
    real_xy = detect.getRealCoord(cx, cy)
    print('real_x: ' + str(real_xy[0]))
    print('real_y: ' + str(real_xy[1]))

    os.chdir(output_dir)
    cv2.imwrite("object_detection.png", copy)