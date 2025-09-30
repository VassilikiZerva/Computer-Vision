import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

filename = next(iter(uploaded))
img = cv2.imread(filename)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_img = gray_img / 255.0

ksize = 31  
sigma = 5.0  
theta_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]  
lamda = 10.0 
gamma = 0.5  
psi = 0 

plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, theta in enumerate(theta_list):
      gabor_kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_64F)
    
    kernel_normalized = (gabor_kernel - gabor_kernel.min()) / (gabor_kernel.max() - gabor_kernel.min())
    
    filtered_img = cv2.filter2D(gray_img, cv2.CV_64F, gabor_kernel)
    
    filtered_img_normalized = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min())
    
    plt.subplot(3, 3, i+2)
    plt.imshow(kernel_normalized, cmap='gray')
    plt.title(f'Gabor Kernel θ={theta:.2f}')
    plt.axis('off')
    
    plt.subplot(3, 3, i+6)
    plt.imshow(filtered_img_normalized, cmap='gray')
    plt.title(f'Filtered Image θ={theta:.2fs}')
    plt.axis('off')

plt.tight_layout()
plt.show() 