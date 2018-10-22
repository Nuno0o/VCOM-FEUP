from cv2 import *
from numpy import *
from matplotlib import pyplot as plt

img = imread("img.png", 0)

img2 = None

img2 = equalizeHist(img)

imshow("1", img)
imshow("2", img2)

waitKey(5000)

hist, bins = histogram(img.flatten(), 256, [0,256])
hist2, bins2 = histogram(img2.flatten(), 256, [0,256])

plot = plt.plot(hist)
plot2 = plt.plot(hist2)
plt.show()