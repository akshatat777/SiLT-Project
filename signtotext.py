import numpy as np
import cv2
import random
from sign_recog_cnn import SignRecogCNN
import torch
from data_processing import normalize

# images = np.load("data/images.npy")
# labels = np.load("data/labels.npy")
# print(images.shape)

# indx = [random.randint(0, 5000) for i in range(10)]
# images = images[indx]
# for i in range(10):
#     cv2.imshow('image',images[i])
#     cv2.waitKey(0)
# images = normalize(images)
# print(images.shape)
# labels = labels[indx]
# alphabet = "abcdefghijklmnopqrstuvwxyz"
# for i in range(10):
#     print(alphabet[labels[i]], end="")
#     #cv2.imshow('image', i)
#     #cv2.waitKey(0)
# print()

# model = SignRecogCNN()
# model.load_state_dict(torch.load('sign_recogn_cnn_larger', map_location=torch.device('cpu')))
# model.eval()
# with torch.no_grad():
#     images = torch.tensor(images).to('cpu')
#     encodings = []
#     preds = model(images)

# converts each sign to appropriate text.
def sign_to_text(sign_encodings):
    """ Converts a list of signs () 
        and returns their corresponding text."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    ls_str = []
    for encoding in sign_encodings:
        ls_str.append(alphabet[np.argmax(encoding)])
    return ls_str

# print(sign_to_text(preds))