from cv2 import *
import camera
'''from tkinter import *

def StartGUI():
    root = Tk()
    root.title("VCOM-2018")
    root.minsize(width=200,height=200)
    root.mainloop()'''

def Menu():
    print('--------- HAND GESTURE DETECTION TOOL ----------\n')
    print('\tPick source image:\n\tImport image - Press 1\n\tTake picture - Press 2\n\t\t')
    option = input('\t>')
    if option == '1':
        img = camera.CaptureCameraImage()
        return img
    else:
        path = input('\tImage Path: ')
        img = imread(path)
        return img
    return null