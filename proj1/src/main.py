import algorithm
import camera
import gui
from cv2 import *

#img = camera.CaptureCameraImage()

img = imread('../imgs/one.jpg')
img = algorithm.Gaussian(img)
img = algorithm.SmoothImage(img)
img = algorithm.ConvertToYCbCr(img)
img = algorithm.ConvertToBinary(img)

namedWindow("Binary")
imshow("Binary",img)
waitKey(5000)
#gui.StartGUI()