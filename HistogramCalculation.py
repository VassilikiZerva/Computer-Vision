#Assignment1
#Histogram Calculation 

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from matplotlib import pyplot as plt

#Upload image
uploaded = files.upload()
filename = next(iter(uploaded))
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)  # Load as grayscale
print("Original Grayscale Image")
cv2_imshow(img)

#Structuring Element (Rectangle)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

#Erosion and Dilation
eroded = cv.erode(img, kernel)
dilated = cv.dilate(img, kernel)

#Results
print("Eroded Image")
cv2_imshow(eroded)
print("Dilated Image")
cv2_imshow(dilated)

#Histograms
def show_histogram(image, label):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist, label=label)

plt.figure(figsize=(10, 5))
show_histogram(img, "Original")
show_histogram(eroded, "Eroded")
show_histogram(dilated, "Dilated")
plt.title("Grayscale Histograms")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show() 
