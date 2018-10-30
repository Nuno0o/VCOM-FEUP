import algorithm
import camera
import gui
from cv2 import *
from matplotlib import pyplot as plt

img = gui.Menu()
img = algorithm.ResizeImage(img)
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToHSV(img)
img = algorithm.SmoothImage(img)
img = algorithm.DetectHands(img)
img2 = algorithm.DetectGestures(img)
imshow("Skin",img)
imshow("Contours",img2)
plt.show()

#img2 = algorithm.ErodeImg(img)
#imshow("Eroded", img2)

#x1,y1,x2,y2 = algorithm.GetRectEdges(img)
#tl = [x1,y1]
#br = [x2,y2]
#algorithm.GetRectSection(img,tl,br)
#print(x1, " ", y1, " ", x2, " ", y2) 
waitKey(0)