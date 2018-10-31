import algorithm
import parseargs
from cv2 import *
from matplotlib import pyplot as plt
import sys

img = parseargs.parseArgs()
if img is None:
    print('Image not found')
    quit()
img = algorithm.ResizeImage(img)
img = algorithm.NormalizeLight(img)
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToHSV(img)
imgs = algorithm.DetectHands(img)
for i in range(0, len(imgs)):
    img2 = algorithm.DetectGestures(imgs[i])
    imshow("Contours"+str(i),img2)
#imshow("Skin",img)
#plt.show()

#img2 = algorithm.ErodeImg(img)
#imshow("Eroded", img2)

#x1,y1,x2,y2 = algorithm.GetRectEdges(img)
#tl = [x1,y1]
#br = [x2,y2]
#algorithm.GetRectSection(img,tl,br)
#print(x1, " ", y1, " ", x2, " ", y2) 
sys.stdout.flush() 
waitKey(0)