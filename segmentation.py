import numpy as np
import cv2
import os

path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

img = cv2.imread(path + '/photo2.png')

# Apply blur filter to "smoothen" the edges
img = cv2.GaussianBlur(img, (7, 7), 0)
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

    for j in range(i + 1, len(horizontal_contours)):
        c2 = horizontal_contours[j]
        approx2 = cv2.approxPolyDP(c2, 0.009 * cv2.arcLength(c, True), True)
        n2 = approx2.ravel()
        y2 = n2[1]

        # if the y value of two lines are within 80 pixels of each other,
        # remove the lower line
        if abs(y2 - y) < 80:
            if (y > y2):
                cv2.drawContours(edges, [c], -1, (0, 0, 0), 2)
            else:
                cv2.drawContours(edges, [c2], -1, (0, 0, 0), 2)
            print(" ")
            break

os.chdir(output_dir)
cv2.imwrite("blue.png", B)
cv2.imwrite("green.png", G)
cv2.imwrite("mask.png", mask)
cv2.imwrite("edges.png", edges)
cv2.imwrite("no_throw.png", no_throw)
cv2.imwrite("red.png", R)
# cv2.imwrite("test.png", lines)
