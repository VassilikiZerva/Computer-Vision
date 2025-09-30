#Assignment1
#Morphological Operations 

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow    
from google.colab import files

#Upload image
uploaded = files.upload()  
filename = next(iter(uploaded))
img = cv.imread(filename, cv.IMREAD_UNCHANGED)  

#Show Original Image
print("Original Image:")
cv2_imshow(img)

#Structuring Elements
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

#Erosion and Dilation
erosion = cv.erode(img, kernel_rect)
dilation = cv.dilate(img, kernel_rect)
print("Erosion:")
cv2_imshow(erosion)
print("Dilation:")
cv2_imshow(dilation)

#Opening and Closing
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_ellipse)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_cross)
print("Opening:")
cv2_imshow(opening)
print("Closing:")
cv2_imshow(closing)

#Gradient, Tophat, Blackhat
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_rect)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel_ellipse)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel_cross)
print("Gradient:")
cv2_imshow(gradient)
print("Tophat:")
cv2_imshow(tophat)
print("Blackhat:")
cv2_imshow(blackhat)

#Sequences: Dilation -> Erosion, Opening -> Closing, Closing -> Opening
dilate_erode = cv.erode(cv.dilate(img, kernel_rect), kernel_rect)
open_close = cv.morphologyEx(cv.morphologyEx(img, cv.MORPH_OPEN, kernel_rect), cv.MORPH_CLOSE, kernel_rect)
close_open = cv.morphologyEx(cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_rect), cv.MORPH_OPEN, kernel_rect)
print("Dilation -> Erosion:")
cv2_imshow(dilate_erode)
print("Opening -> Closing:")
cv2_imshow(open_close)
print("Closing -> Opening:")
cv2_imshow(close_open)

#Distance Transform 
if len(img.shape) == 2 :
    _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    dist_transform = cv.distanceTransform(binary, cv.DIST_L2, 5)
    dist_display = cv.normalize(dist_transform, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    cv2_imshow(dist_display)
else:
    print("Distance Transform skipped: Not a binary image (0 and 255).") 
