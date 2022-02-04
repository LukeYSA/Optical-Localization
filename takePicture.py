import numpy as np
import cv2
import os

input_dir = r'/Users/lukeyin/Desktop/School/Mirror/Research/Optical-Localization/input_img'

def capturePic(a : int = 1) -> None:
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            os.chdir(input_dir)
            cv2.imwrite("photo2.png", frame)
            print("written!")

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    capturePic()
