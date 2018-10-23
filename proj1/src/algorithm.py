from cv2 import *

def SmoothImage(img):
    blur = bilateralFilter(img,5,75,75)
    return blur

def ConvertToYCbCr(img):
    ycbcr = cvtColor(img, COLOR_BGR2YCR_CB)
    return ycbcr

def ConvertToBinary(img):
    y,cb,cr = split(img)
    _,binary = threshold(y,127,255,THRESH_BINARY_INV + THRESH_OTSU)
    return binary

