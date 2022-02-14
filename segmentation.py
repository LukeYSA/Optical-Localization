import numpy as np
import cv2
import os

path = r'/Users/lukeyin/Desktop/Research/Optical-Localization/input_img'
output_dir = r'/Users/lukeyin/Desktop/Research/Optical-Localization/output_img'

img = cv2.imread(path + '/photo2.png')
img = cv2.GaussianBlur(img, (7, 7), 0)
# img = cv2.blur(img, (7, 7))

B, G, R = cv2.split(img)
ret, mask = cv2.threshold(G, 66, 255, cv2.THRESH_BINARY)
# mask = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

edges = cv2.Canny(mask, 100, 200)

os.chdir(output_dir)
cv2.imwrite("blue.png", B)
cv2.imwrite("green.png", G)
cv2.imwrite("mask.png", mask)
cv2.imwrite("edges.png", edges)
cv2.imwrite("red.png", R)
