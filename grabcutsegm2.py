from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

uploaded = files.upload()

for filename in uploaded.keys():
    image_stream = io.BytesIO(uploaded[filename])
    pil_image = Image.open(image_stream).convert('RGB') 
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

mask[100:300, 100:300] = cv2.GC_FGD

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

output_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

result = image * output_mask[:, :, np.newaxis]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("GrabCut Result")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show() 