#Assignment5 - Segmentation using Background Subtraction
#Vasiliki Zerva 

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Loading of the video
video_path = 'static_camera.webm'
cap = cv2.VideoCapture(video_path)

#It reads the first frame to use it
ret, frame = cap.read()
if not ret:
    raise Exception("Can't read the video.")

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
height, width = gray_frame.shape
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Accumulated Weighted Image
accum_weight = 0.01
accum_background = np.float32(gray_frame)

#MOG2 Background Subtractor 
mog2 = cv2.createBackgroundSubtractorMOG2()

#Custom Moving Average Background
custom_background = np.zeros_like(gray_frame, dtype=np.float32)

#Store foregrounds for the chosen frame
chosen_frame_index = num_frames // 2  
foregrounds = {}

#Process video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Accumulate Weighted
    cv2.accumulateWeighted(gray, accum_background, accum_weight)
    fg1 = cv2.absdiff(gray, cv2.convertScaleAbs(accum_background))

    #MOG2
    fg2 = mog2.apply(frame)

    #Custom Moving Average
    custom_background = (custom_background * i + gray) / (i + 1)
    fg3 = cv2.absdiff(gray, cv2.convertScaleAbs(custom_background))

    #Save foregrounds at the chosen frame
    if i == chosen_frame_index:
        foregrounds['frame'] = frame
        foregrounds['fg1'] = fg1
        foregrounds['fg2'] = fg2
        foregrounds['fg3'] = fg3

#Final background models
final_bg1 = cv2.convertScaleAbs(accum_background)
final_bg2 = mog2.getBackgroundImage()
final_bg3 = cv2.convertScaleAbs(custom_background)

cap.release()

#Visualization
def show_image(title, img, cmap='gray'):
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')

#Background
show_image("Background - Accumulate Weighted", final_bg1)
show_image("Background - MOG2", final_bg2)
show_image("Background - Moving Average", final_bg3)

#Original Frame and Foreground masks
show_image("Original Frame", cv2.cvtColor(foregrounds['frame'], cv2.COLOR_BGR2RGB), cmap=None)
show_image("Foreground - Accumulate Weighted", foregrounds['fg1'])
show_image("Foreground - MOG2", foregrounds['fg2'])
show_image("Foreground - Moving Average", foregrounds['fg3'])

plt.show() 