import numpy as np
import cv2

def captureVideo(a : int = 1) -> None:
    vid = cv2.VideoCapture(0)

    while (True):
        ret, frame = vid.read()

        frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)

        img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # white mask
        lower_white = np.array([0,0,168])
        upper_white = np.array([172,111,255])
        white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        white_mask = cv2.bitwise_not(white_mask)

        cv2.imshow('mask', white_mask)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    captureVideo()
