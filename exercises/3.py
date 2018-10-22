from cv2 import *
from numpy import *

cap = VideoCapture(0)
imageee = None
while(True):
    ret, frame = cap.read()

    gray = cvtColor(frame, COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    key = waitKey(1)
    if key & 0xFF == ord('q'):
        imageee = frame
        break



cap.release()
destroyAllWindows()