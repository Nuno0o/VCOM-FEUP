from cv2 import *
import camera

def Menu():
    print('--------- HAND GESTURE DETECTION TOOL ----------\n')
    print('\tPick source image:\n\tTake picture - Press 1\n\tImport image - Press 2\n\t\t')
    option = raw_input('\t>')
    option = str(option)
    if option == '1':
        img = camera.CaptureCameraImage()
        return img
    elif option == '2':
        path = raw_input('\tImage Path: ')
        path = str(path)
        img = imread(path)
        return img
    return null