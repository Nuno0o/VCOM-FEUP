import algorithm
import camera
import gui
from cv2 import *

#img = camera.CaptureCameraImage()

img = imread('../imgs/two.png')
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToYCbCr(img)
img = algorithm.ConvertToBinary(img)
x1,y1,x2,y2 = algorithm.GetRectEdges(img)
algorithm.GetRectSection(img,(x1,y1),(x2,y2))
print(x1, " ", y1, " ", x2, " ", y2) 

namedWindow("Binary")
imshow("Binary",img)
waitKey(0)
#gui.StartGUI()