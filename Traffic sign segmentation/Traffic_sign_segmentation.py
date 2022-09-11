#!/usr/bin/env python

import argparse
import os
import cv2 as cv
import numpy as np


# Change to directory that contain images file
os.chdir("C:/Users/HP/Desktop/Mini project/OneDrive_1_7-21-2022/test")


# Argument parser with a default input image
parser = argparse.ArgumentParser(description = "Traffic sign color segmentation")
parser.add_argument('--input', help = "path to input image", default = '000_0001.png')
args = parser.parse_args()


# Flood fill function for each color channel

# Flood fill for red color channel mask
def rfill(rflood):
   cv.floodFill(rflood, mask, (0, 0), (255, 255, 255));
   rflood_inv = cv.bitwise_not(rflood)
   return rth| rflood_inv


# Flood fill for blue color channel mask
def bfill(bflood):
   cv.floodFill(bflood, mask, (0, 0), (255, 255, 255));
   bflood_inv = cv.bitwise_not(bflood)
   return bth | bflood_inv


# Flood fill for yellow color channel mask
def yfill(yflood):
   cv.floodFill(yflood, mask, (0, 0), (255, 255, 255));
   yflood_inv = cv.bitwise_not(yflood)
   return yth | yflood_inv


# Function to calculate the IoU
def bb_intersection_over_union(boxA, boxB):
# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# Define structuring element for morphological transform
struc_ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
struc_ele2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (39, 39))
struc_ele3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (85, 85))


# Read in images
img = cv.imread(cv.samples.findFile(args.input))


# Height and width of the original image
height = img.shape[0]
width = img.shape[1]


# Resize the image to constant size (500x500)
img = cv.resize(img, (500, 500), interpolation = cv.INTER_AREA)
img_resize = img.copy()


# Perform Bilateral filtering to remove noise
img_blur = cv.bilateralFilter(img, 15, 75, 75)


# Convert image to YUV color space
img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2YUV)


# Perform histogram equalization
img_blur[:,:,0] = cv.equalizeHist(img_blur[:, :, 0])
img_eq = cv.cvtColor(img_blur, cv.COLOR_YUV2BGR)


# Convert image to HSV color space
img_hsv = cv.cvtColor(img_eq, cv.COLOR_BGR2HSV)


# Assure/ verify that the hsv color range that we determined
lower_blue = (90, 30, 50)
upper_blue = (128, 255, 255)

lower_red1 = (0, 50, 70)
upper_red1 = (9, 255, 255)

lower_red2 = (159, 50, 70)
upper_red2 = (180, 255, 255)

lower_yellow = (10, 70, 50)
upper_yellow = (38, 255, 255)


# Perform color segmentation
bmask = cv.inRange(img_hsv, lower_blue, upper_blue )
    
r1mask = cv.inRange(img_hsv, lower_red1, upper_red1)
r2mask = cv.inRange(img_hsv, lower_red2, upper_red2)
rmask=r1mask+r2mask
    
ymask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
    
rimg = img_resize.copy()
bimg = img_resize.copy()
yimg = img_resize.copy()

final_r = cv.bitwise_and(rimg, img_resize, mask = rmask)
final_b = cv.bitwise_and(bimg, img_resize, mask = bmask)
final_y = cv.bitwise_and(yimg, img_resize, mask = ymask)


# Convert image to GRAY
red = cv.cvtColor(final_r, cv.COLOR_BGR2GRAY)   
blue = cv.cvtColor(final_b, cv.COLOR_BGR2GRAY)   
yellow = cv.cvtColor(final_y, cv.COLOR_BGR2GRAY)


# Perform Otsu Thresholding
ret, rth = cv.threshold(red, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret, bth = cv.threshold(blue, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret, yth = cv.threshold(yellow, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)


# Draw a boarder to the image
rth = cv.rectangle(rth, (0, 0), (499, 499), 0, 10, cv.LINE_AA)
bth = cv.rectangle(bth, (0, 0), (499, 499), 0, 10, cv.LINE_AA)
yth = cv.rectangle(yth, (0, 0), (499, 499), 0, 10, cv.LINE_AA)


# Flood fill for red color channel
rth = cv.cvtColor(rth, cv.COLOR_GRAY2BGR)
rflood = rth.copy()
h, w = rth.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
rflood_res = rfill(rflood)


# Flood fill for blue color channel
bth = cv.cvtColor(bth, cv.COLOR_GRAY2BGR)
bflood = bth.copy()
h, w = bth.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
bflood_res = bfill(bflood)


# Flood fill for yellow color channel
yth = cv.cvtColor(yth, cv.COLOR_GRAY2BGR)
yflood = yth.copy()
h, w = yth.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
yflood_res = yfill(yflood)

red_g = cv.cvtColor(rflood_res, cv.COLOR_BGR2GRAY)   
blue_g = cv.cvtColor(bflood_res, cv.COLOR_BGR2GRAY)   
yellow_g = cv.cvtColor(yflood_res, cv.COLOR_BGR2GRAY)   


# Perform Morphological Transformations
# Closing
rclosing = cv.morphologyEx(red_g, cv.MORPH_CLOSE, struc_ele)
bclosing = cv.morphologyEx(blue_g, cv.MORPH_CLOSE, struc_ele)
yclosing = cv.morphologyEx(yellow_g, cv.MORPH_CLOSE, struc_ele)


# Opening
ropening = cv.morphologyEx(rclosing, cv.MORPH_OPEN, struc_ele2)
bopening = cv.morphologyEx(bclosing, cv.MORPH_OPEN, struc_ele2)
yopening = cv.morphologyEx(yclosing, cv.MORPH_OPEN, struc_ele2)

rtest = ropening.copy()
btest = bopening.copy()
ytest = yopening.copy()


# Find contours
rcontour, hierarchy = cv.findContours(ropening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
bcontour, hierarchy = cv.findContours(bopening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
ycontour, hierarchy = cv.findContours(yopening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# If there is any countour found
# Find the largest countour among all the contours found for each color
if len(rcontour) != 0:
    r = max(rcontour, key = cv.contourArea)
if len(bcontour) != 0:
    b = max(bcontour, key = cv.contourArea)
if len(ycontour) != 0:
    y = max(ycontour, key = cv.contourArea)


# If the countour area is not equal to contour area of largest contour or smaller than 15000
# Remove that contour         
for k in rcontour:
    if (cv.contourArea(k) != cv.contourArea(r) or cv.contourArea(k) < 15000):
        cv.drawContours(ropening, [k], 0, (0, 0, 0), -1)
for k in bcontour:
    if (cv.contourArea(k) != cv.contourArea(b) or cv.contourArea(k) < 15000):
            cv.drawContours(bopening, [k], 0, (0, 0, 0), -1)
for k in ycontour:
     if (cv.contourArea(k) != cv.contourArea(y) or cv.contourArea(k) < 15000):
            cv.drawContours(yopening, [k], 0, (0, 0, 0), -1)

# Dilate the yeloow color mask
yopening = cv.morphologyEx(yopening, cv.MORPH_DILATE, struc_ele3)

rcontour, hierarchy = cv.findContours(ropening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
bcontour, hierarchy = cv.findContours(bopening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
ycontour, hierarchy = cv.findContours(yopening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


# Find the array length of the contour point    
if len(rcontour) != 0:
    rcnt = len(rcontour[0])
else:
    rcnt = 99999
        
if len(bcontour) != 0:
    bcnt = len(bcontour[0])
else:
    bcnt = 99999
        
if len(ycontour) != 0:
    ycnt = len(ycontour[0])
else:
    ycnt = 99999    


# Find the shortest contour array length among red, blue and yellow mask as the final mask
if (rcnt < bcnt):
    if(rcnt < ycnt):
        fmask = ropening
    else:
        fmask = yopening
else:
    if(bcnt < ycnt):
        fmask = bopening
    else:
        fmask = yopening

# If no mask created draw the whole image as bounding box
if((rcnt == bcnt) and (bcnt == ycnt) and (ycnt == 99999)):
    y1 = 0
    x1 = 0
    y2 = 499
    x2 = 499
    seg = cv.rectangle(img_resize, (y1, x1), (y2, x2), 0, 2, cv.LINE_AA) 
    
else:
    # Use final mask to segment the traffic sign
    # Draw the bounding box
    where = np.array(np.where(fmask))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    seg = cv.rectangle(img_resize, (y1, x1), (y2, x2), 0, 2, cv.LINE_AA)


# Read the annotation file
file_read = open("TsignRecgTrain4170Annotation.txt", "r")
#file_read = open("TsignRecgTest1994Annotation.txt", "r")

text = cv.samples.findFile(args.input)
  
lines = file_read.readlines()


# Search and get the bounding box information
for line in lines:
    if text in line:
        info = line.split(";")
        break

# Create x and y scaling factor
yscale = 500/height
xscale = 500/width


# Scale the bounding box value 
ya = int(int(info[3])*yscale)
yb = int(int(info[5])*xscale)

xa = int(int(info[4])*xscale)
xb = int(int(info[6])*yscale)


# Eeturn IoU value of the image
print(bb_intersection_over_union([x1, y1, x2, y2], [xa, ya, xb, yb]))


# Display the segmented result
cv.imshow('image', seg)
cv.waitKey(0)
cv.destroyAllWindows()


