from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
import math

np.set_printoptions(threshold=np.nan)

def SmoothImage(img):
    blur = GaussianBlur(img,(5,5),0)
    return blur

def ResizeImage(img):
    img2 = resize(img, (400,400))
    return img2

def ConvertToYCrCb(img):
    ycrcb = cvtColor(img, COLOR_BGR2YCR_CB)
    return ycrcb

def ConvertToHSV(img):
    hsv = cvtColor(img, COLOR_BGR2HSV)
    return hsv

def EdgeDetection(img):
    edge = Canny(img,120,140)
    return edge

def DetectHands(img):
    #for human hands, HSV =~ (0-50, 40-180, 50-255)
    x_min = 0
    x_max = 30
    y_min = 20
    y_max = 173
    z_min = 40
    z_max = 255
    ranges = inRange(img, np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max]))

    #apply median blur to remove small components
    ranges = medianBlur(ranges, ksize=9)

    #find small components not caught by median blur
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(ranges, connectivity=4)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    #remove background
    nb_components = nb_components - 1
    stats2 = stats[1:]
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 150
    #your answer image
    img2 = np.zeros((output.shape),np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if stats2[i][4] >= min_size:
            img2[output == i + 1] = 255

    return ranges

def DetectGestures(img):
    _, contours, hierarchy = findContours(img, RETR_TREE, CHAIN_APPROX_SIMPLE)
    hull = []
    color_centroids = (255,255,255)
    color_peaks = (0,0,255)
    color_contours = (0,255,0)
    color = (255,0,0)

    for i in range(len(contours)):
        hull.append(RemoveRepeatedPoints(convexHull(contours[i],False)))
    print(hull)
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    biggest_centroid, bc_index = GetBiggestCentroid(hull)

    if biggest_centroid == []:
        return img
    
    peaks = GetPeaks(biggest_centroid, hull[bc_index])

    # Draw Centroid
    drawing[biggest_centroid[1]][biggest_centroid[0]] = color_centroids
    # To see it slightly better, draw pixels around it in same color
    drawing[biggest_centroid[1] + 1][biggest_centroid[0]] = color_centroids
    drawing[biggest_centroid[1] - 1][biggest_centroid[0]] = color_centroids
    drawing[biggest_centroid[1] + 1][biggest_centroid[0] + 1] = color_centroids
    drawing[biggest_centroid[1] + 1][biggest_centroid[0] - 1] = color_centroids
    drawing[biggest_centroid[1] - 1][biggest_centroid[0] + 1] = color_centroids
    drawing[biggest_centroid[1] - 1][biggest_centroid[0] - 1] = color_centroids

    drawContours(drawing,contours,bc_index,color_contours,1,8,hierarchy)
    drawContours(drawing,hull,bc_index,color,1,8)

    for i in range(len(peaks)):
        drawing[peaks[i][1]][peaks[i][0]] = color_peaks

    return drawing

def RemoveRepeatedPoints(hull):
    if len(hull) == 1:
        return hull
    groups = []
    current_group = []
    max_dist = 20
    for i in range(0, len(hull)):
        if len(current_group) == 0:
            current_group.append(hull[i])
            continue
        currx = hull[i][0][0]
        curry = hull[i][0][1]
        latestx = current_group[len(current_group)-1][0][0]
        latesty = current_group[len(current_group)-1][0][1]
        #do manhattan distance to check if points are close
        if abs(abs(currx - latestx) + abs(curry - latesty)) < max_dist:
            current_group.append(hull[i])
        #discard close points and keep only the one in the center
        else:
            groups.append(current_group[math.floor(len(current_group)/2)])
            current_group = []
            current_group.append(hull[i])
    if len(current_group) > 0:
        currx = hull[0][0][0]
        curry = hull[0][0][1]
        latestx = current_group[len(current_group)-1][0][0]
        latesty = current_group[len(current_group)-1][0][1]
        if len(groups) > 0 and abs(abs(currx - latestx) + abs(curry - latesty)) >= max_dist:
            groups.append(current_group[math.floor(len(current_group)/2)])
        elif len(groups) == 0:
            groups.append(current_group[math.floor(len(current_group)/2)])
    return np.array(groups, dtype=np.int32)

def GetCentroid(hull):
    centroids = []
    for h in hull:
        x = []
        y = []
        for point in h:
            x.append(point[0][0])
            y.append(point[0][1])
        centroids.append((int(round(np.mean(x))), int(round(np.mean(y)))))
    return centroids

# Gets biggest centroid based on average distance to vertices
# --> ONLY RELEVANT IF ASSUMING HAND AS BIGGEST CENTROID
def GetBiggestCentroid(hull):
    centroids = GetCentroid(hull)
    max_avg = 0
    biggest_centroid = []
    max_i = 0
    for i in range(len(centroids)):
        dist_x = 0
        dist_y = 0
        x_centroid = centroids[i][0]
        y_centroid = centroids[i][1]
        for point in hull[i]:
            dist_x += abs(x_centroid - point[0][0])
            dist_y += abs(y_centroid - point[0][1])
        cur_avg = (dist_x / len(hull[i]) + dist_y / len(hull[i])) / 2
        if cur_avg > max_avg:
            max_avg = cur_avg
            biggest_centroid = [x_centroid, y_centroid]
            max_i = i
    return biggest_centroid, max_i

# Given the centroid and the hull, calculate peaks and filter irrelevant peaks (below 25% of dist between centroid and the biggest peak)
# centroid = [x_centroid, y_centroid]
# points = [ [[x_point1, y_point1]], [[x_point2, y_point2]], ...]
def GetPeaks(centroid, points):
    peaks = []
    # Get max VERTICAL peak
    max_peak = 0
    for i in range(len(points)):
        if points[i][0][1] < centroid[1]:
            if points[i][0][1] > max_peak:
                max_peak = abs(points[i][0][1] - centroid[1])
    
    for i in range(len(points)):
        if (centroid[1] - points[i][0][1]) >= 0.75 * max_peak:
            peaks.append(points[i][0])

    GetPeakPlot(centroid, peaks)

    return peaks

def GetPeakPlot(centroid, peaks):
    N = len(peaks)
    peaksx = []
    peaksy = []

    for i in range(N):
        peaksx.append(peaks[i][0])
        peaksy.append(peaks[i][1])
    
    colors = (0,0,0)
    area = np.pi*3

    plt.scatter(peaksx, peaksy, s=area, c=colors, alpha=0.5)
    plt.title('Scatter Plot of Peaks')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    #plt.show()

    return centroid

def ErodeImg(img):
    kernel1Size_x = 2
    kernel1Size_y = 2
    kernel2Size_x = 3
    kernel2Size_y = 3
    kernel = np.ones((kernel1Size_x,kernel1Size_y),np.uint8)
    kernel2 = np.ones((kernel2Size_x,kernel2Size_y),np.uint8)
    #erosion = erode(img, kernel, iterations=3)
    #erosion = dilate(erosion,kernel2, iterations=4)

    return erosion

def ConvertToBinary(img):
    y,cb,cr = split(img)
    _,binary = threshold(y,127,255,THRESH_BINARY_INV + THRESH_OTSU)
    return binary

def SweepTopBottom(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(0,height):
        for x in range(0,width):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepBottomTop(img):
    height, width = img.shape[0], img.shape[1]
    for y in range(height-1,-1,-1):
        for x in range(0,width):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepLeftRight(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(0,width):
        for y in range(0,height):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def SweepRightLeft(img):
    height, width = img.shape[0], img.shape[1]
    for x in range(width-1,-1,-1):
        for y in range(0,height):
            if img[y][x] == 255:
                return x,y
    return -1,-1

def GetRectEdges(img):
    x1,_ = SweepLeftRight(img)
    x2,_ = SweepRightLeft(img)
    _,y3 = SweepTopBottom(img)
    _,y4 = SweepBottomTop(img)
    topleft_x = x1
    topleft_y = y3
    bottomright_x = x2
    bottomright_y = y4
    return topleft_x, topleft_y, bottomright_x, bottomright_y

def GetRectSection(img,tl,br):
    #rectangle(img, (x1,y1), (x2,y2), 128, 1)
    clone = img.copy()
    roi = clone[tl[1]:br[1],tl[0]:br[0]]
    imshow("cropped rectangle", roi)

def NormalizeLight(img):
    '''avg_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    avg_color[0] = avg_color[0]*0.2
    avg_color[1] = avg_color[1]
    avg_color[2] = avg_color[2]*0.2
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            img[y][x] = img[y][x]-avg_color
    imshow("redness", img)'''
    normalizedimg = np.zeros((400,400))
    normalizedimg = normalize(img, normalizedimg, 50 , 190, NORM_MINMAX)
    return normalizedimg


