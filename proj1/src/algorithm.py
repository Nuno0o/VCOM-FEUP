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

def SweepLeftRight(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(0,height):
        for x in range(0,width):
            if img[y][x] == 255:
                break
    return x,y

def SweepRightLeft(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(0,height):
        for x in range(width-1,0,-1):
            if img[y][x] == 255:
                break
    return x,y
    
def SweepTopBottom(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(0,width):
        for y in range(0,height):
            if img[y][x] == 255:
                break
    return x,y

def SweepBottomTop(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(0,width):
        for y in range(height-1,0,-1):
            if img[y][x] == 255:
                break
    return x,y