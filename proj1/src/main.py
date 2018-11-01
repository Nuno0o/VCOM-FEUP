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
if len(imgs) == 0:
    print('No hands detected')
for i in range(0, len(imgs)):
    x1,y1,x2,y2 = algorithm.GetRectEdges(imgs[i])
    tl = [x1,y1]
    br = [x2,y2]
    segment = algorithm.GetRectSection(imgs[i], tl, br)
    hand, fingers, thumb = algorithm.DetectGestures(segment)
    strfingers = 'finger' if fingers == 1 else 'fingers'
    strthumb = 'a raised thumb' if thumb else 'no thumb'
    print('Hand ' + str(i) + ' with ' + str(fingers) + ' ' + strfingers + ' and ' + strthumb)
    imshow('Hand ' + str(i),hand)
sys.stdout.flush() 
waitKey(0)