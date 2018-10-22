import algorithm
import camera
import gui
from cv2 import *

img = camera.CaptureCameraImage()
img = algorithm.SmoothImage(img)

namedWindow("xd")
imshow("xd",img)
waitKey(5000)
#gui.StartGUI()