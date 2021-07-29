import cv2
from data_processing import resize_crop
import numpy as np

def crop_hand(img, hands):
    # margin gives some space between the tips of fingers and the bounding box (bbox) measured in pixels
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if not results.multi_hand_landmarks:
        return None
    results = np.array([[[lm.x, lm.y, lm.z] for lm in hand_lms.landmark] for hand_lms in results.multi_hand_landmarks])
    del imgRGB, img
    return results[None,...].astype(np.float32)