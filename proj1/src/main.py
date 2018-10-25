import algorithm
import camera
import gui
from cv2 import *

img = gui.Menu()
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToHSV(img)
#img = algorithm.ConvertToBinary(img)
img = algorithm.DetectSkin(img)

img2 = algorithm.ErodeImg(img)
#x1,y1,x2,y2 = algorithm.GetRectEdges(img)
#tl = [x1,y1]
#br = [x2,y2]
#algorithm.GetRectSection(img,tl,br)
#print(x1, " ", y1, " ", x2, " ", y2) 

imshow("Binary",img)
imshow("TRinary", img2)
waitKey(0)
#gui.StartGUI()