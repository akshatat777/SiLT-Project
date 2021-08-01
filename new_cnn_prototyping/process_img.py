import cv2
import numpy as np
import os
import mediapipe as mp
from img_proc_help import resize_crop,crop_hand_cnn

data = []
hands = mp.solutions.hands.Hands(static_image_mode=True,
								max_num_hands=1,
								min_detection_confidence=0.5)
for C in os.listdir('data'):
	if 'DS' in C or '.npy' in C:
		continue
	for img_path in os.listdir(f'data/{C}'):
		if 'DS' in img_path:
			continue
		img = cv2.imread(f'data/{C}/{img_path}')
		img = crop_hand_cnn(img,hands)
		if img is None:
			continue
		label = ord(C)-ord('A')
		data.append((img[0],label))

np.random.shuffle(data)

train_len = int(len(data)*0.8)

train_img, train_lab = [], []
test_img, test_lab = [], []
for i, (img, label) in enumerate(data):
	if i < train_len:
		train_img.append(img)
		train_lab.append(label)
	else:
		test_img.append(img)
		test_lab.append(label)

train_img = np.array(train_img)
train_lab = np.array(train_lab)
test_img = np.array(test_img)
test_lab = np.array(test_lab)
np.save('data/train_img', train_img)
np.save('data/train_lab', train_lab)
np.save('data/test_img', test_img)
np.save('data/test_lab', test_lab)


