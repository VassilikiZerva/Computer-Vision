#Assignment5 - Dense Optical Flow
#Vasiliki Zerva

import cv2
import numpy as np
from google.colab.patches import cv2_imshow


#Load video
video_path = 'static_camera.webm'
cap = cv2.VideoCapture(video_path)

#Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error loading video")
    cap.release()
    import sys
    sys.exit() #for clean exit

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0 #Initialize a frame counter
display_interval = 30 #Display a frame every 30 frames 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #Calculate magnitude and angle of flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    #Threshold motion (magnitude)
    #Ensure mag is not None before thresholding
    if mag is not None:
      motion_mask = cv2.threshold(mag, 1.0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    else:
      #If mag is None, skip processing this frame
      prev_gray = gray.copy()
      frame_count += 1
      continue

    #Clean noise
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    #Find contours
    #Ensure motion_mask is not None before finding contours
    if motion_mask is not None:
      contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[:2]
    else:
       prev_gray = gray.copy()
       frame_count += 1
       continue


    #Green rectangles
    #Ensure frame is not None before drawing
    if frame is not None:
      for cnt in contours:
          if cv2.contourArea(cnt) > 500:  #To filter small movements
              x, y, w, h = cv2.boundingRect(cnt)
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Show result using cv2_imshow 
    if frame_count % display_interval == 0 and frame is not None:
        print(f"Displaying frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
        cv2_imshow(frame)
        print("-" * 20) #Separator

    prev_gray = gray.copy()
    frame_count += 1

cap.release()
cv2.destroyAllWindows()