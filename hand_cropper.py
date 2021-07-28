import cv2
import mediapipe as mp
from img_proc_helper import resize_crop

def crop_hand(cap, mode=False, margin=100, normalized_size = 100):
    # margin gives some space between the tips of fingers and the bounding box (bbox) measured in pixels
    hands = mp.solutions.hands.Hands(static_image_mode=mode,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.45)
    # plays recording from camera and processes each image
    success, img = cap.read()
    if not success:
        print('unable to read')
        return []
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    cropped_results = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmark_listx = []
            landmark_listy = []
            for lm in handLms.landmark:
                h, w, c = img.shape
                # here we multiply by the width and height because the landmarks are auto-normalized based on
                # the width and height of the displayed image
                landmark_listx.append((lm.x*w))
                landmark_listy.append((lm.y*h))
            end = (int(max(landmark_listx))+margin, int(max(landmark_listy))+margin)
            start = (int(min(landmark_listx))-margin, int(min(landmark_listy))-margin)
            cropped_img = img[start[1] : end[1], start[0] : end[0]]
            if cropped_img.shape[0] <= 0 or cropped_img.shape[1] <= 0:
                continue
            cropped_results.append(resize_crop(cropped_img))
    return cropped_results