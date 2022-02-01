import cv2

def captureVideo(a : int = 1) -> None:
    vid = cv2.VideoCapture(1)

    while (True):
        ret, frame = vid.read()

        frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)


        edge_detect = cv2.Canny(frame, 100, 200)
        cv2.imshow('Edge detect', edge_detect)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    captureVideo()
