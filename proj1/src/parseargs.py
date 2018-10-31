import argparse
import image
from cv2 import *

parser = argparse.ArgumentParser(description="Detect number of fingers in hands from image file or camera")
parser.add_argument('method' ,help='\'camera\' or \'file\'', type=str)
parser.add_argument('-i', '--image', dest='path', default='img.jpg',type=str)
def parseArgs():
    args = parser.parse_args()
    arg = args.method
    if arg == 'camera':
        return image.CaptureCameraImage()
    elif arg == 'file':
        print('Opening image ' + args.path)
        return image.ReadImageFile(args.path)