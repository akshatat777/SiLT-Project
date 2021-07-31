from sign_recogn_cnn import SignRecogCNN
import torch
from torch import nn
import numpy as np
from img_proc_help import normalize, resize_crop
import cv2
import os

def read_batch(batch_size : int = 64, train : bool = True):
    if train:
        total_x, total_y = np.load('data/train_img.npy'), np.load('data/train_lab.npy')
    else:
        total_x, total_y = np.load('data/test_img.npy'), np.load('data/test_lab.npy')
    indices = np.arange(len(total_y))
    np.random.shuffle(indices)
    for batch_i in range(len(total_y)//batch_size):
        idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
        yield normalize(total_x[idx]), total_y[idx]

def data_aug(images):
	# N,3,224,224
	shifth = np.random.randint(-35,35)
	shiftv = np.random.randint(-35,35)
	images = np.roll(images,shifth,axis=-2)
	images = np.roll(images,shiftv,axis=-1)
	cshift = np.random.uniform(-0.3,0.3)
	images += cshift
	images = np.clip(images, 0, 1)
	return images

def read_test():
	imgs = []
	labs = []
	for k in os.listdir('test'):
		if 'DS' in k:
			continue
		img = cv2.imread(f'test/{k}')
		imgs.append(resize_crop(img))
		labs.append(ord(k.split('.')[0])-ord('a'))
	return torch.tensor(normalize(imgs)).to('cpu'), torch.tensor(labs).to('cpu')

model = SignRecogCNN().to('cpu')
optimizer = torch.optim.Adam(model.parameters(),lr=20*1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(950):
	print(f'training epoch {epoch}')
	train_gen = read_batch()
	for i, (imgs, labs) in enumerate(train_gen):
		imgs = data_aug(imgs)
		imgs = torch.tensor(imgs).to('cpu')
		labs = torch.tensor(labs).to('cpu')
		preds = model(imgs)
		# N, 26
		loss = loss_fn(preds, labs)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		with torch.no_grad():
			acc = torch.mean((torch.argmax(preds.detach(), dim=1) == labs).float()).detach().item()
			print('train loss,acc: ',loss.detach().item(), acc)
			if (i+1) % 10 == 0:
				model.eval()
				test_imgs, test_labs = read_test()
				preds = model(test_imgs)
				loss = loss_fn(preds, test_labs).detach().item()
				acc = torch.mean((torch.argmax(preds.detach(), dim=1) == test_labs).float()).detach().item()
				print('val loss,acc: ',loss,acc)
				model.train()
