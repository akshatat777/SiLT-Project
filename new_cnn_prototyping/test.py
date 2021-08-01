import cv2
import mediapipe as mp
import os
from img_proc_help import normalize, crop_hand_cnn
import torch
from torch import nn
from sign_recogn_cnn import SignRecogCNN

def read_test():
	hands = mp.solutions.hands.Hands(static_image_mode=True,
								max_num_hands=1,
								min_detection_confidence=0.5)
	imgs = []
	labs = []
	for k in os.listdir('test'):
		if 'DS' in k:
			continue
		img = cv2.imread(f'test/{k}')
		img = crop_hand_cnn(img,hands)
		imgs.append(img[0])
		labs.append(ord(k.split('.')[0])-ord('a'))
	return torch.tensor(normalize(imgs)).to('cpu'), torch.tensor(labs).to('cpu')

imgs, labs = read_test()

imgs = torch.tensor(imgs).to('cpu')
labs = torch.tensor(labs).to('cpu')

model = SignRecogCNN()
model.load_state_dict(torch.load('sign_recogn_cnn',map_location=torch.device('cpu')))
loss_fn = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
	preds = model(imgs)
	loss = loss_fn(preds, labs)
	acc = torch.mean((torch.argmax(preds.detach(), dim=1) == labs).float()).detach().item()
	print(loss,acc)
