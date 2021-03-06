import numpy as np
import cv2
import os

"""
*** IMPORTANT ***
Change these paths to the absolute paths in your context
"""
# Path to your /input_img directory
path = r'<insert path>'
# Path to your /output_img directory
output_dir = r'<insert path>'

"""
This is a testing program to play around with image processing and colors
"""

img=cv2.imread(path + '/bob.png')
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower red mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper red mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my red masks
red_mask = mask0+mask1
red_mask = cv2.bitwise_not(red_mask)

# green mask
lower_green = np.array([36,25,25])
upper_green = np.array([86,255,255])
green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
green_mask = cv2.bitwise_not(green_mask)

# white mask
lower_white = np.array([0,0,200])
upper_white = np.array([145,60,255])
white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
white_mask = cv2.bitwise_not(white_mask)

# set my output img to zero everywhere except my mask
# output_img = img.copy()
# output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
# output_hsv = img_hsv.copy()
# output_hsv[np.where(mask==0)] = 0

os.chdir(output_dir)
cv2.imwrite('red_mask.png', red_mask)
cv2.imwrite('green_mask.png', green_mask)
cv2.imwrite('white_mask.png', white_mask)
