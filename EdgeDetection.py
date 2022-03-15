import numpy as np
import cv2
import os
from collections import defaultdict

path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

def drawFull(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Apply blur filter to "smoothen" the edges
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.blur(img, (7, 7))

    # Split gray scale images on blue, green, and red
    B, G, R = cv2.split(img)

    # Threshold the green gray scale image so we can extract just the green parts
    ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny filter so we can extract the edges
    # edges = cv2.Canny(mask, 7, 7)

    # Sobel filter version
    grad_x = cv2.Sobel(mask, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(mask, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_xy = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    abs_grad_xy = cv2.convertScaleAbs(grad_xy)

    os.chdir(output_dir)
    cv2.imwrite("grad_x.png", abs_grad_x)
    cv2.imwrite("grad_y.png", abs_grad_y)
    cv2.imwrite("grad_xy.png", abs_grad_xy)
    
    edges = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)
    # edges = cv2.Sobel(mask, ddepth, 1, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    return edges

def segment_lines(lines, k=2, **kwargs):
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

def intersection(line1, line2):
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
    return [[x0], [y0]]

def segmented_intersection(lines):
    intersections = []
    # Compare lines in each group
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    return intersections

def makeMatrix(intersections):
    res = np.matrix()


def plot_points(img, points):
    copy = img.copy()
    for point in points:
        # print(str(point[0][0]) + ", " + str(point[1][0]))
        copy = cv2.circle(copy, (point[0][0], point[1][0]), radius=5, color=(255, 0, 0), thickness=-1)
    return copy

if __name__ == '__main__':
    img = cv2.imread(path + '/chess.png')

    # horizontal_edges = drawHorizontal(img)
    # vertical_edges = drawVertical(img)
    # res = horizontal_edges + vertical_edges

    res = drawFull(img)

    # detect_cross(img)
    rho, theta, thresh = 1, np.pi/180, 150
    lines = cv2.HoughLines(res, rho, theta, thresh)
    segmented = segment_lines(lines)
    intersections = segmented_intersection(segmented)

    print(intersections)
    points = plot_points(res, intersections)

    os.chdir(output_dir)
    # cv2.imwrite("hedges.png", horizontal_edges)
    # cv2.imwrite("vedges.png", vertical_edges)
    cv2.imwrite("res.png", res)
    cv2.imwrite("points.png", points)
