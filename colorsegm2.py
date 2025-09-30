import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from google.colab.patches import cv2_imshow  

from google.colab import files
uploaded = files.upload()  

filename = next(iter(uploaded))

image = cv2.imread(filename)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()


def adaptive_threshold(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    adaptive_gaussian = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(adaptive_mean, cmap='gray')
    plt.title('Adaptive Mean Thresholding')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(adaptive_gaussian, cmap='gray')
    plt.title('Adaptive Gaussian Thresholding')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return adaptive_mean, adaptive_gaussian

def otsu_threshold(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    ret1, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    ret2, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(simple_thresh, cmap='gray')
    plt.title(f'Simple Thresholding (value={ret1})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title(f'Otsu Thresholding (value={ret2})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.axvline(x=ret2, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Histogram with Otsu Threshold (value={ret2})')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    return otsu_thresh, ret2

def lab_delta_segmentation(image, x_start, y_start, x_end, y_end):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    target_region = lab_image[y_start:y_end, x_start:x_end]
    
    target_lab = np.mean(target_region, axis=(0, 1))
    print(f"Target LAB values: L={target_lab[0]}, a={target_lab[1]}, b={target_lab[2]}")
    
    target_color_lab = np.ones((100, 100, 3), dtype=np.uint8)
    target_color_lab[:, :] = target_lab
    target_color_bgr = cv2.cvtColor(target_color_lab, cv2.COLOR_LAB2BGR)
    target_color_rgb = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2RGB)
    
    L, a, b = lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]
    L_diff = L.astype(np.float32) - target_lab[0]
    a_diff = a.astype(np.float32) - target_lab[1]
    b_diff = b.astype(np.float32) - target_lab[2]
    
    delta_e = np.sqrt(L_diff**2 + a_diff**2 + b_diff**2)
    
    delta_e_normalized = (delta_e / delta_e.max() * 255).astype(np.uint8)
    
    thresholds = [15, 30, 45]
    segmented_results = []
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(target_color_rgb)
    plt.title('Target Color')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(delta_e_normalized, cmap='jet')
    plt.title('Delta E (Color Difference)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    for i, threshold in enumerate(thresholds):
        _, segmented = cv2.threshold(delta_e_normalized, threshold, 255, cv2.THRESH_BINARY_INV)
        segmented_results.append(segmented)
        
        plt.subplot(234 + i)
        plt.imshow(segmented, cmap='gray')
        plt.title(f'Segmented (Threshold = {threshold})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return delta_e_normalized, segmented_results

def kmeans_segmentation(image, k=3):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    segmented_image = centers[labels.flatten()]
    
    segmented_image = segmented_image.reshape(image.shape)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f'K-means Segmentation (k={k})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 4*((k+2)//3)))
    plt.subplot(k+1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    for i in range(k):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == i] = 255
        mask = mask.reshape(image.shape[0], image.shape[1])
        
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        color_sample = np.ones((50, 50, 3), dtype=np.uint8) * centers[i].reshape(1, 1, 3)
        
        plt.subplot(k+1, 3, i+2)
        plt.imshow(cv2.cvtColor(color_sample, cv2.COLOR_BGR2RGB))
        plt.title(f'Cluster {i+1} Color')
        plt.axis('off')
        
        plt.subplot(k+1, 3, i+k+2)
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Cluster {i+1} Segment')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return segmented_image, labels.reshape(image.shape[0], image.shape[1])

def run_all_segmentation_methods(image):
    print("1.Adaptive Thresholding")
    adaptive_mean, adaptive_gaussian = adaptive_threshold(image)
    
    print("\n2.Otsu Thresholding")
    otsu_result, otsu_threshold_value = otsu_threshold(image)
    
    print("\n3.Lab Delta Segmentation")
    print("\nSlected region")

    h, w = image.shape[:2]
    x_center, y_center = w // 2, h // 2
    region_size = min(h, w) // 10
    
    x_start = x_center - region_size // 2
    y_start = y_center - region_size // 2
    x_end = x_center + region_size // 2
    y_end = y_center + region_size // 2
    
    region_image = image.copy()
    cv2.rectangle(region_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))
    plt.title('Selected Region for Target Color')
    plt.axis('off')
    plt.show()
    
    delta_e, segmented_results = lab_delta_segmentation(image, x_start, y_start, x_end, y_end)
    
    print("\n4.K-means Segmentation")
    for k in [3, 5]:
        print(f"K-means with k={k}")
        kmeans_result, kmeans_labels = kmeans_segmentation(image, k)

run_all_segmentation_methods(image) 