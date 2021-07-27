from cv2 import data
import numpy as np
import cv2
import os
import random

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
            database.append((image,path))

random.shuffle(database)
imgs = np.array([img for img, lab in database])
labels = np.array([ord(label)-ord('a') for img, label in database])
np.save('data/images', imgs)
np.save('data/labels', labels)
