import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from google.colab.patches import cv2_imshow
import math

print("Please upload 3 different hand gesture images:")
uploaded_images = files.upload()

gestures = []
gesture_names = []
hu_moments_list = []

for filename in uploaded_images.keys():

    image = cv2.imdecode(np.frombuffer(uploaded_images[filename], np.uint8), cv2.IMREAD_COLOR)
    
    name = filename.split('.')[0]
    gesture_names.append(name)
    gestures.append(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        moments = cv2.moments(mask)
        hu_moments = cv2.HuMoments(moments)
        
        for i in range(7):
            if hu_moments[i] != 0:
                hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
                
        hu_moments_list.append(hu_moments.flatten())
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Original - {name}')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title('Contour')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(mask, cmap='gray')
        plt.title('Final Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Hu Moments for {name}:")
        for i, moment in enumerate(hu_moments.flatten()):
            print(f"h{i+1}: {moment}")
        print("\n")

def compare_moments(unknown_moments, reference_moments, method="euclidean"):
    """Compare Hu moments using different distance metrics"""
    if method == "euclidean":
        return np.sqrt(np.sum((unknown_moments - reference_moments) ** 2))
    elif method == "chi-square":
        
        sum_val = 0
        for i in range(len(unknown_moments)):
            if unknown_moments[i] != 0 or reference_moments[i] != 0:
                sum_val += ((unknown_moments[i] - reference_moments[i]) ** 2) / (unknown_moments[i] + reference_moments[i] + 1e-10)
        return 0.5 * sum_val
    elif method == "cosine":
        dot_product = np.dot(unknown_moments, reference_moments)
        norm_unknown = np.linalg.norm(unknown_moments)
        norm_reference = np.linalg.norm(reference_moments)

        if norm_unknown * norm_reference == 0:
            return float('inf')
        return 1 - (dot_product / (norm_unknown * norm_reference))
    else:
        raise ValueError(f"Unknown distance method: {method}")
