import algorithm
import camera
import gui
from cv2 import *

img = gui.Menu()
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToYCbCr(img)
img = algorithm.ConvertToBinary(img)
x1,y1,x2,y2 = algorithm.GetRectEdges(img)
tl = [x1,y1]
br = [x2,y2]
algorithm.GetRectSection(img,tl,br)
print(x1, " ", y1, " ", x2, " ", y2) 

namedWindow("Binary")
imshow("Binary",img)
waitKey(0)
#gui.StartGUI()