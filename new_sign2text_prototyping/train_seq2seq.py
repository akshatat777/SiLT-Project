from seq2seq import Seq2Seq
from process_text import gen_batch
import torch
from torch import nn
import numpy as np

model = Seq2Seq().to('cpu')
batch_gen = gen_batch()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=1e-4)
for i, (batch_vecs, batch_lets) in enumerate(batch_gen):
	# (T,N,29)
	# (N,T')
	batch_lets = np.transpose(batch_lets)
	# (T',N)
	batch_vecs = torch.tensor(batch_vecs).to('cpu')
	batch_lets = torch.tensor(batch_lets).to('cpu')
	preds = model(batch_vecs.float())
	# (T,N,29)
	batch_lets = torch.flatten(batch_lets)
	preds = torch.flatten(preds,start_dim=0,end_dim=1)[:len(batch_lets)]
	loss = loss_fn(preds,batch_lets.long())

	optim.zero_grad()
	loss.backward()
	optim.step()

	with torch.no_grad():
		print(f'batch {i}, loss: {loss.detach().item()}')
		acc = torch.mean((torch.argmax(preds.detach(), dim=1) == batch_lets.detach()).float())
		print(f'batch {i}, acc: {acc}')
