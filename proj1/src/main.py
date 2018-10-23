import algorithm
import camera
import gui
from cv2 import *

#img = camera.CaptureCameraImage()

img = imread('../imgs/two.png')
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToYCbCr(img)
img = algorithm.ConvertToBinary(img)
x1,y1 = algorithm.SweepLeftRight(img)
x2,y2 = algorithm.SweepRightLeft(img)
x3,y3 = algorithm.SweepTopBottom(img)
x4,y4 = algorithm.SweepBottomTop(img)
print(x1,y1)
print(x2,y2)
print(x3,y3)
print(x4,y4)  

namedWindow("Binary")
imshow("Binary",img)
waitKey(0)
#gui.StartGUI()