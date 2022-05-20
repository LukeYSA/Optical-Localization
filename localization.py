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
    detect = EdgeDetection(img, output_dir=output_dir)

    # Detect and draw the edges of the image
    res = detect.drawFull()

    # Group edges into two groups based on their theta
    segmented = detect.segment_lines(res)

    # Detect intersection of lines and output the point of intersection
    intersections = detect.segmented_intersection(segmented)
    # print(intersections)

    intersection_matrix = detect.construct_matrix(intersections)
    # print(intersection_matrix)

    # Plot the intersection points on the image
    points = detect.plot_points(res, intersection_matrix)
    # print(intersection_matrix)

    # Fill the position matrix (the mapping from pixel coordinate to real life coordinate)
    position_matrix = detect.fill_position_matrix(intersection_matrix)
    # detect.userTest(position_matrix)

    # Initialize object detection process
    object = ObjectDetection(img, output_dir=output_dir)
    cx, cy = object.findObjectCenter()
    print('cx: ' + str(cx))
    print('cy: ' + str(cy))
    copy = img.copy()
    copy = cv2.circle(copy, (cx, cy), radius=7, color=(255, 0, 0), thickness=-1)
    real_xy = position_matrix[cy][cx]
    print('real_x: ' + str(real_xy[0]))
    print('real_y: ' + str(real_xy[1]))
    # print('real y: ' + str(real_xy[1]))

    os.chdir(output_dir)
    # cv2.imwrite("hedges.png", horizontal_edges)
    # cv2.imwrite("vedges.png", vertical_edges)
    cv2.imwrite("res.png", res)
    cv2.imwrite("points.png", points)
    cv2.imwrite("object_detection.png", copy)