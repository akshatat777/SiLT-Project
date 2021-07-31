import cv2
import numpy as np
import os
from img_proc_help import resize_crop

data = []
for C in os.listdir('data'):
	if 'DS' in C:
		continue
	for img_path in os.listdir(f'data/{C}'):
		if 'DS' in img_path:
			continue
		img = resize_crop(cv2.imread(f'data/{C}/{img_path}'))
		label = ord(C)-ord('A')
		data.append((img,label))

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


