import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import io

uploaded = files.upload()

for filename in uploaded:
    image_stream = io.BytesIO(uploaded[filename])
    pil_image = Image.open(image_stream).convert("RGB")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

image = cv2.resize(image, (400, 400))

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, algorithm=cv2.ximgproc.SLICO, region_size=10, ruler=10.0)
slic.iterate(10)
mask_slic = slic.getLabelContourMask()
labels_slic = slic.getLabels()

slic_result = image.copy()
slic_result[mask_slic == 255] = [0, 0, 255] 

plt.figure(figsize=(8, 8))
plt.title("SLIC Superpixels")
plt.imshow(cv2.cvtColor(slic_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()  