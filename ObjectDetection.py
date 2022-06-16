import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple

"""
*** IMPORTANT ***
Change these paths to the absolute paths in your context
"""
# Path to your /input_img directory
path = r'<insert path>'
# Path to your /output_img directory
output_dir = r'<insert path>'

"""
- This class handles the detection of the object and the finding of its center pixel.
- This class should be initialized whenever there is a new frame from the video, and
    call findObjectCenter() to find the pixel coordinate of the object's center, if
    there is any.
"""
class ObjectDetection:
    def __init__(self, image, output_dir = None) -> None:
        """
        Call this construxtor every time there is a new frame from the video.
        """
        self.image = image
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = None

    def findObjectCenter(self) -> Tuple[int, int]:
        """Get the pixel coordinate of the object, if there is any.
        
        Returns:
            A Tuple of (x pixel coord, y pixel coord)
        """
        # Use the blue image and find object contour
        B, G, R = cv2.split(self.image)
        ret, mask = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
        img_copy = self.image.copy()
        cv2.drawContours(img_copy, contours, -1, (0,255,0), 7)

        os.chdir(self.output_dir)
        cv2.imwrite("object_blue.png", B)
        cv2.imwrite("object_mask.png", mask)
        cv2.imwrite("object_contours.png", img_copy)

        # Chooses the first contour that it sees
        cnt = contours[0]

        # Get the moments of the shape
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return (cx, cy)