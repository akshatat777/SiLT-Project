import cv2
import numpy as np
from data_processing import resize

def crop_hand_joint(img, hands):
    # margin gives some space between the tips of fingers and the bounding box (bbox) measured in pixels
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if not results.multi_hand_landmarks:
        return None
    results = np.array([[[lm.x, lm.y, lm.z] for lm in hand_lms.landmark] for hand_lms in results.multi_hand_landmarks])
    del imgRGB, img
    return results[None,...].astype(np.float32)

def crop_hand_cnn(img, hands, margin=0.2):
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    cropped_results = []
    if not results.multi_hand_landmarks:
    	return None
    for handLms in results.multi_hand_landmarks:
        landmark_listx = []
        landmark_listy = []
        for lm in handLms.landmark:
            h, w, c = img.shape
            # here we multiply by the width and height because the landmarks are auto-normalized based on
            # the width and height of the displayed image
            landmark_listx.append((lm.x*w))
            landmark_listy.append((lm.y*h))
        end = (int(max(landmark_listx))+int(margin*w), int(max(landmark_listy))+int(margin*h))
        start = (int(min(landmark_listx))-int(margin*w), int(min(landmark_listy))-int(margin*h))
        cropped_img = img[start[1] : end[1], start[0] : end[0]]
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        	continue
        print(cropped_img.shape)
        cropped_results.append(resize(cropped_img))
    if len(cropped_results) == 0:
    	return None
    return np.array(cropped_results,dtype = np.float32)