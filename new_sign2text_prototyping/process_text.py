import re
from seq2seq import Seq2Seq
import numpy as np

def read_wiki():
	path_to_wikipedia = "wikipedia2text-extracted.txt"  # update this path if necessary
	with open(path_to_wikipedia, "rb") as f:
		wikipedia = f.read().decode().lower()
	wikipedia = re.sub(r'[^a-z ]+', '', wikipedia)
	return wikipedia

def gen_data():
	wiki = read_wiki().split()
	idx = np.arange(len(wiki))
	np.random.shuffle(idx)
	for i in idx:
		length = min(np.random.randint(1,5),len(wiki)-i)
		text = ' '.join(wiki[i:i+length])
		t_text = []
		for let in text:
			rep = np.random.randint(5,40)
			t_text.extend([let]*rep)
		vecs = []

		def let2num(let):
			if let == ' ':
				return 26
			else:
				return ord(let)-ord('a')

		for let in t_text:
			letvec = np.zeros(29)
			letvec[let2num(let)]=1
			vecs.append(letvec)
		labels = [let2num(let) for let in text]
		labels.append(28)
		yield vecs, labels

def gen_batch(batch_size = 32):
	data_gen = gen_data()
	batch_vecs = []
	batch_lets = []
	max_len = 0
	max_let_len = 0
	end_let = np.zeros((29))
	end_let[28] = 1.0
	for i,(vecs,lets) in enumerate(data_gen):
		#print(vecs[0].shape)
		batch_vecs.append(vecs)
		batch_lets.append(lets)
		max_let_len = max(max_let_len, len(lets))
		max_len = max(max_len,len(vecs))
		if (i+1) % batch_size == 0:
			max_len += 1
			for j in range(len(batch_vecs)):
				batch_vecs[j].append(end_let)
				batch_vecs[j].extend([end_let]*(max_len - len(batch_vecs[j])))
				batch_vecs[j] = np.array(batch_vecs[j])
				#print(batch_vecs[j].shape)
				batch_vecs[j] = batch_vecs[j][:,None,:]
			batch_vecs = np.concatenate(batch_vecs,axis=1)
			batch_lets = [lets + [28]*(max_let_len-len(lets)) for lets in batch_lets] 
			yield batch_vecs, np.array(batch_lets,dtype=np.int8)
			batch_vecs = []
			batch_lets = []
			max_let_len = 0
			max_len = 0