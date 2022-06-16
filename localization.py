import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple

from EdgeDetection import EdgeDetection
from ObjectDetection import ObjectDetection

"""
This is the localization program, run this to perform localization of a object.
It currently takes an image and detects the object and prints its pixel coordinate (cx, cy)
and its real life coordinate (real_x, real_y)
"""
if __name__ == '__main__':
    """
    *** IMPORTANT ***
    Change these paths to the absolute paths in your context
    """
    # Path to your /input_img directory
    path = r'<insert path>'
    # Path to your /output_img directory
    output_dir = r'<insert path>'

    img = cv2.imread(path + '/chess_irl.png')
    # Initialize edge detection and pixel coordinate to real coordinate mapping
    detect = EdgeDetection(img, output_dir=output_dir)

    # Initialize object detection process
    object = ObjectDetection(img, output_dir=output_dir)
    # Finds the center of the object
    cx, cy = object.findObjectCenter()
    print('cx: ' + str(cx))
    print('cy: ' + str(cy))
    copy = img.copy()

    # Draw a circle on the detected object
    copy = cv2.circle(copy, (cx, cy), radius=7, color=(255, 0, 0), thickness=-1)
    # Get the real life xy-coordinate
    real_xy = detect.getRealCoord(cx, cy)
    print('real_x: ' + str(real_xy[0]))
    print('real_y: ' + str(real_xy[1]))

    # Save the circle image to check if detection was successful
    os.chdir(output_dir)
    cv2.imwrite("object_detection.png", copy)