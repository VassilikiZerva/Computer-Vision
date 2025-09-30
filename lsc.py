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

region_size = 10  
ratio = 0.075    

lsc = cv2.ximgproc.createSuperpixelLSC(image, region_size=region_size, ratio=ratio)

lsc.iterate(10)

labels = lsc.getLabels()
mask_lsc = lsc.getLabelContourMask()

lsc_result = image.copy()
lsc_result[mask_lsc == 255] = [0, 0, 255] 

plt.figure(figsize=(8, 8))
plt.title("LSC Superpixels")
plt.imshow(cv2.cvtColor(lsc_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show() 