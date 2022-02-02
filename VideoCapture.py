import numpy as np
import cv2

def captureVideo(a : int = 1) -> None:
    vid = cv2.VideoCapture(0)

    while (True):
        ret, frame = vid.read()

        frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)

        # edge_detect = cv2.Canny(frame, 100, 200)
        # cv2.imshow('Edge detect', edge_detect)

        result = frame.copy()
        bresult = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([155,25,0])
        upper = np.array([179,255,255])

        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

        mask = cv2.inRange(frame, lower, upper)
        bmask = cv2.inRange(frame, lower_blue, upper_blue)
        result = cv2.bitwise_and(result, result, mask=mask)
        bresult = cv2.bitwise_and(bresult, bresult, mask=bmask)

        # cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        cv2.imshow('bresult', bresult)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    captureVideo()
