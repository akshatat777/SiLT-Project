from cv2 import data
import numpy as np
import cv2
import os
import random
from data_processing import resize_crop

chars = ['A','B','C','D','E']
database = []
for c in chars:
    for path in os.listdir(f'data/{c}'):
        if path == '.DS_Store':
            continue
        for img_path in os.listdir(f'data/{c}/{path}'):
            if img_path == '.DS_Store':
                continue
            image = cv2.imread(f'data/{c}/{path}/{img_path}')
            # scales/pads the images so that they're all shape (100, 100, 3):
            image = resize_crop(image)
            database.append((image,ord(path)-ord('a')))

random.shuffle(database)

imgs = []
labs = []
t_imgs = []
t_labs = []
train_split = 0.8
for i, (img, lab) in enumerate(database):
    if img.shape != (100,100,3):
        print('abnormal shape')
        print(img.shape)
    if i > train_split*len(database):
        t_imgs.append(img)
        t_labs.append(lab)
    else:
        imgs.append(img)
        labs.append(lab)
imgs = np.array(imgs)
labels = np.array(labs)
t_imgs = np.array(t_imgs)
t_labels = np.array(t_labs)
np.save('data/images', imgs)
np.save('data/labels', labels)
np.save('data/test_images', t_imgs)
np.save('data/test_labels', t_labels)
