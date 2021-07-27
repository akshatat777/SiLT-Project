from os import read
from data_processing import read_batch_train

for train_x_batch, truth in read_batch_train(64):
    print(train_x_batch.shape)