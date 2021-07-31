from seq2seq import Seq2Seq
from process_text import gen_batch

model = Seq2Seq()
batch_gen = gen_batch()
for batch_vecs, batch_lets in batch_gen:
	print(batch_vecs.shape)
	print(batch_lets.shape)