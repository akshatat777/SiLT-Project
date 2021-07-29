import cv2
from hand_cropper import crop_hand
import gc
import mediapipe as mp

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.45)
while True:
	suc, img = cap.read()
	if not suc:
		continue
	results = crop_hand(img,hands)
	cv2.imshow('image',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
	del img, results
	gc.collect()