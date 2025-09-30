from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Upload the Large Picture:")
uploaded_big = files.upload()
big_img_path = next(iter(uploaded_big))
scene_img = cv2.imread(big_img_path, cv2.IMREAD_GRAYSCALE)

print("Upload the Patch :")
uploaded_patch = files.upload()
patch_img_path = next(iter(uploaded_patch))
patch_img = cv2.imread(patch_img_path, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(patch_img, None)
kp2, des2 = sift.detectAndCompute(scene_img, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = patch_img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    scene_img_color = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)
    result_img = cv2.polylines(scene_img_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    plt.title('Patch localized with SIFT + SIFT')
    plt.axis('off')
    plt.show()
else:
    print("No matches found.") 