from cv2 import *
from numpy import *

img = imread('imgs.jpg')

'''img2 = cvtColor(img, COLOR_RGB2GRAY)

saltpepper = zeros((len(img2),len(img2[0])), uint8)

randu(saltpepper, 0, 255)

for y in range(0, len(img2)):
    for x in range(0, len(img2[0])):
        if(saltpepper[y][x] > 229):
            img2[y][x] = 255
        if(saltpepper[y][x] < 26):
            img2[y][x] = 0
'''


'''imgR = zeros((len(img),len(img[0])), uint8)
imgG = zeros((len(img),len(img[0])), uint8)
imgB = zeros((len(img),len(img[0])), uint8)
for y in range(0, len(img)):
    for x in range(0, len(img[0])):
        imgR[y][x] = img[y,x,2]
        imgG[y][x] = img[y,x,1]
        imgB[y][x] = img[y,x,0]

namedWindow("window1")
namedWindow("window2")
namedWindow("window3")

imshow("window1", imgR)
imshow("window2", imgG)
imshow("window3", imgB)'''


waitKey(0)