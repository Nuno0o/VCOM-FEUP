from cv2 import *

def SmoothImage(img):
    blur = bilateralFilter(img,5,75,75)
    return blur