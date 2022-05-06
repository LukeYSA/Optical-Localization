import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple

path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

class EdgeDetection:
    def __init__(self, image, output_dir = None) -> None:
        self.image = image
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = None

    def drawFull(self):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        # Apply blur filter to "smoothen" the edges
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = cv2.blur(img, (7, 7))

        # Split gray scale images on blue, green, and red
        B, G, R = cv2.split(self.image)

        # Threshold the green gray scale image so we can extract just the green parts
        ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
        # mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Apply Canny filter so we can extract the edges
        # edges = cv2.Canny(mask, 7, 7)

        # Sobel filter version
        grad_x = cv2.Sobel(mask, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(mask, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        if output_dir is not None:
            os.chdir(output_dir)
            cv2.imwrite("edge_grad_x.png", abs_grad_x)
            cv2.imwrite("edge_grad_y.png", abs_grad_y)
        
        edges = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)
        # edges = cv2.Sobel(mask, ddepth, 1, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        return edges

    def segment_lines(self, edgeImage, k=2, **kwargs):
        rho, theta, thresh = 1, np.pi/180, 170
        lines = cv2.HoughLines(edgeImage, rho, theta, thresh)

        # Set parameters
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flag', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)

        # Get the angles in radians
        angles = np.array([line[0][1] for line in lines])
        # Get unit circle coordinates of the angle
        pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)

        # Group the points based on where they are on the unit circle
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)

        # Group the lines based on the kmeans grouping
        segmented = defaultdict(list)
        for i, line in enumerate(lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())
        return segmented

    def intersection(self, line1, line2):
        # Find intersection point of two lines from rho and theta
        # Solve:
        # x * cos(theta1) + y * sin(theta1) = r1
        # x * cos(theta2) + y * sin(theta2) = r1
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        # x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [int(np.round(x0)), int(np.round(y0))]

    def segmented_intersection(self, lines):
        intersections = []
        # Compare lines in each group
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(self.intersection(line1, line2))

        intersections.sort()
        return intersections

    def plot_points(self, img, points):
        copy = img.copy()
        for point in points:
            # print(str(point[0]) + ", " + str(point[1]))
            copy = cv2.circle(copy, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(copy, "(x,y): " + str(point[0]) + ", " + str(point[1]),
                        (point[0], point[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return copy

    def construct_matrix(self, intersections):
        # Put intersection points into a matrix based on their relative positions
        n = int(np.sqrt(len(intersections)))
        # print(str(n))
        # Initialize a n*n*2 matrix
        intersection_matrix = [[[0 for k in range(2)] for j in range(n)] for i in range(n)]

        index = 0
        for i in range(n):
            for j in range(n):
                intersection_matrix[i][j] = intersections[index]
                index += 1
                if index >= len(intersections):
                    break

        return intersection_matrix


    """
    This function currently assumes that the grid is perfectly upright.
    Need to explore the case where it is tilted

    *** Maybe straighten the image before processing? ***
    """ 
    def fill_position_matrix(self, intersection_matrix):
        height, width, channels = self.image.shape
        # print(str(height) + ", " + str(width))
        position_matrix = -1 * np.ones((height, width), dtype=object)
        # position_matrix = -1 * np.ones((height, width))

        # For each point in intersection_matrix, look at its immediate right and down
        # Consider y=ax+b, calculate a and b where x and y are the pixel coordinates
        # Calculate distance between two points and get the real world distance between
        # pixels 10mm/dist
        # Write the real world coordinates into the position_matrix
        row = len(intersection_matrix)
        col = len(intersection_matrix[0])

        # General case, right and below of the current dot
        for i in range(0, row - 1):
            for j in range(0, col - 1):
                curr = intersection_matrix[i][j]
                right = intersection_matrix[i + 1][j]
                bot = intersection_matrix[i][j + 1]
                botright = intersection_matrix[i + 1][j + 1]
                # print("curr: " + str(curr) + " right: " + str(right) + " below: " + str(below))
                x1 = curr[0]
                y1 = curr[1]
                x2 = bot[0]
                y2 = bot[1]
                x3 = right[0]
                y3 = right[1]
                x4 = botright[0]
                y4 = botright[1]

                # right (horizontal)
                rightlength = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
                rightm = (y3 - y1) / (x3 - x1)
                rightb = y1

                # bottom right (horizontal)
                botrightlength = np.sqrt((x4 - x2)**2 + (y4 - y2)**2)
                botrightm = (y4 - y2) / (x4 - x2)
                botrightb = y2
                for x in range(x1, x3 + 1):
                    topy = int((rightm * x) + rightb)
                    boty = int((botrightm * x) + botrightb)
                    dist = np.sqrt((x - x1)**2 + (topy - y1)**2)
                    realx = (dist/rightlength) * 10 + (i * 10)

                    for y in range(topy, boty + 1):
                        ydist = topy - y
                        realy = (ydist/(topy - boty)) * 10 + (j * 10)
                        position_matrix[y][x] = [realx, realy]

        return position_matrix

    def userTest(self, position_matrix):
        while True:
            x = int(input("x: "))
            y = int(input("y: "))
            if x == -1:
                break
            print(position_matrix[y][x])

if __name__ == '__main__':
    # Read in the image
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

    os.chdir(output_dir)
    # cv2.imwrite("hedges.png", horizontal_edges)
    # cv2.imwrite("vedges.png", vertical_edges)
    cv2.imwrite("edge_res.png", res)
    cv2.imwrite("edge_points.png", points)
