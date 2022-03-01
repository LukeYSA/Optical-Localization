import numpy as np
import cv2
import os
from collections import defaultdict

path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

def drawHorizontal(img):
    # Apply blur filter to "smoothen" the edges
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = cv2.blur(img, (7, 7))

    # Split gray scale images on blue, green, and red
    B, G, R = cv2.split(img)

    # Threshold the green gray scale image so we can extract just the green parts
    ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny filter so we can extract the edges
    edges = cv2.Canny(mask, 100, 200)

    # remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_contours = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vertical_contours = vertical_contours[0] if len(vertical_contours) == 2 else vertical_contours[1]
    for c in vertical_contours:
        cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)

    # clean up horizontal lines
    # horizontal_contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # horizontal_contours = horizontal_contours[0] if len(horizontal_contours) == 2 else horizontal_contours[1]
    # for c in horizontal_contours:
    #     if cv2.arcLength(c, True) < 50:
    #         cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)

    no_throw = edges.copy()

    # throw away lower lines in the "stripe"
    # also throw away lines that are trash from removing verticals
    # This is always guaranteed to preserve the upper line in each stripe
    # if one match is found, then the other edge (the one discovered in the inner loop)
    # will never find another match because it is guaranteed that the upper edge of the stripe
    # is within a certain distance of the lower edge of the stripe, and nothing else in that range
    horizontal_contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    horizontal_contours = horizontal_contours[0] if len(horizontal_contours) == 2 else horizontal_contours[1]
    for i in range(len(horizontal_contours)):
        c = horizontal_contours[i]

        # if this line is too short, this means that it is trash
        # from removing the vertical lines
        if cv2.arcLength(c, True) < 50:
            cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)
            continue

        # Get the x, y positions
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        n = approx.ravel()
        y = n[1]

        # print(str(y))

        for j in range(i + 1, len(horizontal_contours)):
            c2 = horizontal_contours[j]
            approx2 = cv2.approxPolyDP(c2, 0.009 * cv2.arcLength(c, True), True)
            n2 = approx2.ravel()
            y2 = n2[1]

            # if the y value of two lines are within 80 pixels of each other,
            # remove the lower line
            if abs(y2 - y) < 80 and abs(y2 - y) > 10:
                if (y > y2):
                    cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)
                else:
                    cv2.drawContours(edges, [c2], -1, (0, 0, 0), 2)
                # print(" ")

    # os.chdir(output_dir)
    # cv2.imwrite("blue.png", B)
    # cv2.imwrite("green.png", G)
    # cv2.imwrite("mask.png", mask)
    # cv2.imwrite("edges.png", edges)
    # cv2.imwrite("no_throw.png", no_throw)
    # cv2.imwrite("red.png", R)
    # cv2.imwrite("lines.png", lines)
    return edges

def drawVertical(img):
    # Apply blur filter to "smoothen" the edges
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = cv2.blur(img, (7, 7))

    # Split gray scale images on blue, green, and red
    B, G, R = cv2.split(img)

    # Threshold the green gray scale image so we can extract just the green parts
    ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny filter so we can extract the edges
    edges = cv2.Canny(mask, 100, 200)

    # remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_contours = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    horizontal_contours = horizontal_contours[0] if len(horizontal_contours) == 2 else horizontal_contours[1]
    for c in horizontal_contours:
        cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)

    # clean up horizontal lines
    # horizontal_contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # horizontal_contours = horizontal_contours[0] if len(horizontal_contours) == 2 else horizontal_contours[1]
    # for c in horizontal_contours:
    #     if cv2.arcLength(c, True) < 50:
    #         cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)

    no_throw = edges.copy()

    # throw away lower lines in the "stripe"
    # also throw away lines that are trash from removing verticals
    # This is always guaranteed to preserve the upper line in each stripe
    # if one match is found, then the other edge (the one discovered in the inner loop)
    # will never find another match because it is guaranteed that the upper edge of the stripe
    # is within a certain distance of the lower edge of the stripe, and nothing else in that range
    vertical_contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vertical_contours = vertical_contours[0] if len(vertical_contours) == 2 else vertical_contours[1]
    for i in range(len(vertical_contours)):
        c = vertical_contours[i]

        # if this line is too short, this means that it is trash
        # from removing the vertical lines
        if cv2.arcLength(c, True) < 50:
            cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)
            continue

        # Get the x, y positions
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        n = approx.ravel()
        x = n[0]

        # print(str(y))

        for j in range(i + 1, len(vertical_contours)):
            c2 = vertical_contours[j]
            approx2 = cv2.approxPolyDP(c2, 0.009 * cv2.arcLength(c, True), True)
            n2 = approx2.ravel()
            x2 = n2[0]

            # if the y value of two lines are within 80 pixels of each other,
            # remove the lower line
            if abs(x2 - x) < 80 and abs(x2 - x) > 10:
                if (x > x2):
                    cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)
                else:
                    cv2.drawContours(edges, [c2], -1, (0, 0, 0), 2)
                # print(" ")

    # os.chdir(output_dir)
    # cv2.imwrite("blue.png", B)
    # cv2.imwrite("green.png", G)
    # cv2.imwrite("mask.png", mask)
    # cv2.imwrite("edges.png", edges)
    # cv2.imwrite("no_throw.png", no_throw)
    # cv2.imwrite("red.png", R)
    return edges

def drawFull(img):
    # Apply blur filter to "smoothen" the edges
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = cv2.blur(img, (7, 7))

    # Split gray scale images on blue, green, and red
    B, G, R = cv2.split(img)

    # Threshold the green gray scale image so we can extract just the green parts
    ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny filter so we can extract the edges
    edges = cv2.Canny(mask, 100, 200)

    return edges

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_cross(img):
    # Sliding window to detect the cross pattern
    winW = 3
    winH = 3
    cross = np.matrix([[0, 255, 0], [255, 0, 0], [0, 0, 0]])
    for (x, y, window) in sliding_window(res, stepSize=1, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # Problem: the cross section is not always a perfect cross, it might have gaps
        # How to detect all variations of the cross sections
        if np.array_equal(cross, window):
            print(str(window))
            print(" ")
            print(str(x) + ", " + str(y))
            print(" ")

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
    rho, theta, thresh = 2, np.pi/180, 400
    lines = cv2.HoughLines(res, rho, theta, thresh)
    segmented = segment_lines(lines)
    intersections = segmented_intersection(segmented)

    # print(str(intersections))
    points = plot_points(res, intersections)

    os.chdir(output_dir)
    # cv2.imwrite("hedges.png", horizontal_edges)
    # cv2.imwrite("vedges.png", vertical_edges)
    cv2.imwrite("res.png", res)
    cv2.imwrite("points.png", points)
