from cv2 import *

"""Returns image from camera when Space is pressed, or None when Esc is pressed
"""
def CaptureCameraImage():
    cap = VideoCapture(0)
    img = None
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = waitKey(1)
        space_key = 32
        esc_key = 27
        if key == space_key:
            img = frame
            break
        if key == esc_key:
            break
    cap.release()
    #destroyAllWindows()
    return img