from cv2 import *
from numpy import *
from matplotlib import pyplot as plt

img = imread("noisy.png")

kernel = ones((5,5), float32)/25

mean = filter2D(img,-1,kernel)

gauss = GaussianBlur(img, (5,5),0)

median = medianBlur(img, 5)

bil = bilateralFilter(img, 9, 75,75)

imshow("old", img)
imshow("mean", mean)
imshow("gauss", gauss)
imshow("median", median)
imshow("bil", bil)

waitKey(5000)