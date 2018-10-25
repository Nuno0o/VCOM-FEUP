from cv2 import *
import numpy as np

def SmoothImage(img):
    blur = bilateralFilter(img,5,75,75)
    return blur

def ConvertToYCbCr(img):
    ycbcr = cvtColor(img, COLOR_BGR2YCR_CB)
    return ycbcr

def ConvertToHSV(img):
    hsv = cvtColor(img, COLOR_BGR2HSV)
    return hsv

def DetectSkin(img):
    h_min = 0
    h_max = 50
    s_min = int(round(0.23 * 255))
    s_max = int(round(0.68 * 255))
    ranges = inRange(img, np.array([h_min, s_min, 0]), np.array([h_max, s_max, 255]))
    return ranges

def ErodeImg(img):
    kernel1Size_x = int(round(img.shape[0]/140))
    kernel1Size_y = int(round(img.shape[1]/140))
    kernel2Size_x = int(round(img.shape[0]/110))
    kernel2Size_y = int(round(img.shape[1]/110))
    kernel = np.ones((kernel1Size_x,kernel1Size_y),np.uint8)
    kernel2 = np.ones((kernel2Size_x,kernel2Size_y),np.uint8)
    erosion = erode(img, kernel, iterations=8)
    erosion = dilate(erosion,kernel2, iterations=3)
    erosion = dilate(erosion,kernel, iterations=4)

    return erosion

def ConvertToBinary(img):
    y,cb,cr = split(img)
    _,binary = threshold(y,127,255,THRESH_BINARY_INV + THRESH_OTSU)
    return binary

def SweepTopBottom(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(0,height):
        for x in range(0,width):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepBottomTop(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(height-1,-1,-1):
        for x in range(0,width):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepLeftRight(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(0,width):
        for y in range(0,height):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepRightLeft(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(width-1,-1,-1):
        for y in range(0,height):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def GetRectEdges(img):
    x1,_ = SweepLeftRight(img)
    x2,_ = SweepRightLeft(img)
    _,y3 = SweepTopBottom(img)
    _,y4 = SweepBottomTop(img)
    topleft_x = x1
    topleft_y = y3
    bottomright_x = x2
    bottomright_y = y4
    return topleft_x, topleft_y, bottomright_x, bottomright_y

def GetRectSection(img,tl,br):
    #rectangle(img, (x1,y1), (x2,y2), 128, 1)
    clone = img.copy()
    roi = clone[tl[1]:br[1],tl[0]:br[0]]
    imshow("cropped rectangle", roi)


