import numpy as np
import cv2

img=cv2.imread("3.jpg")
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
upper_green = np.array([70,255,255])
green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
green_mask = cv2.bitwise_not(green_mask)

# white mask
lower_white = np.array([0,0,168])
upper_white = np.array([172,111,255])
white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
white_mask = cv2.bitwise_not(white_mask)

# set my output img to zero everywhere except my mask
# output_img = img.copy()
# output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
# output_hsv = img_hsv.copy()
# output_hsv[np.where(mask==0)] = 0

cv2.imwrite('red_mask.jpg', red_mask)
cv2.imwrite('green_mask.jpg', green_mask)
cv2.imwrite('white_mask.jpg', white_mask)
