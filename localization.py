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

    img = cv2.imread(path + '/chess.png')
    detect = EdgeDetection(img, output_dir=output_dir)

    # Detect and draw the edges of the image
    res = detect.drawFull()

    # Group edges into two groups based on their theta
    segmented = detect.segment_lines(res)

    # Detect intersection of lines and output the point of intersection
    intersections = detect.segmented_intersection(segmented)
    # print(intersections)

    # Plot the intersection points on the image
    points = detect.plot_points(res, intersections)

    intersection_matrix = detect.construct_matrix(intersections)
    # print(intersection_matrix)

    position_matrix = detect.fill_position_matrix(intersection_matrix)
    # plt.imshow(position_matrix, interpolation='none')
    # plt.show()
    detect.userTest(position_matrix)

    # Initialize object detection process
    object = ObjectDetection(img, output_dir=output_dir)
    cx, cy = object.findObjectCenter()

    os.chdir(output_dir)
    # cv2.imwrite("hedges.png", horizontal_edges)
    # cv2.imwrite("vedges.png", vertical_edges)
    cv2.imwrite("res.png", res)
    cv2.imwrite("points.png", points)