import cv2
from torch.functional import norm
from data_processing import normalize
from sign_recog_cnn import SignRecogCNN
import torch
from signtotext import sign_to_text

img = cv2.imread('handtest.png')

cv2.imshow('image', img)

cv2.waitKey(0)

img = normalize(img[None,...])

model = SignRecogCNN()
model.load_state_dict(torch.load('sign_recogn_cnn_larger',map_location=torch.device('cpu')))
model.eval()
print(img.shape)
with torch.no_grad():
    preds = model(torch.tensor(img).to('cpu'))
    print(preds)
    print(sign_to_text(preds))
