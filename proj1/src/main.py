import sys
from cv2 import *
from matplotlib import pyplot as plt
import argparse
import algorithm
import image


parser = argparse.ArgumentParser(description="Detect number of fingers in hands from image file or camera")
parser.add_argument('method' ,help='\'camera\' or \'file\'', type=str)
parser.add_argument('-i', '--image', dest='path', default='img.jpg',type=str)
parser.add_argument('-a', '--advanced', dest='adv', action='store_true')
img = None

args = parser.parse_args()
arg = args.method
if arg == 'camera':
    img = image.CaptureCameraImage()
elif arg == 'file':
    print('Opening image ' + args.path)
    img = image.ReadImageFile(args.path)

if img is None:
    print('Image not found')
    quit()

img = algorithm.ResizeImage(img)
img = algorithm.SmoothImage(img)
imgs = algorithm.DetectHands(img, args.adv)
if len(imgs) == 0:
    print('No hands detected')
for i in range(0, len(imgs)):
    x1,y1,x2,y2 = algorithm.GetRectEdges(imgs[i])
    tl = [x1,y1]
    br = [x2,y2]
    segment = algorithm.GetRectSection(imgs[i], tl, br)
    segment = algorithm.ResizeImage(segment)
    hand, fingers, thumb = algorithm.DetectGestures(segment)
    strfingers = 'finger' if fingers == 1 else 'fingers'
    strthumb = 'a raised thumb' if thumb else 'no thumb'
    print('Hand ' + str(i) + ' with ' + str(fingers) + ' ' + strfingers + ' and ' + strthumb)
    imshow('Hand ' + str(i),hand)
sys.stdout.flush() 
waitKey(0)