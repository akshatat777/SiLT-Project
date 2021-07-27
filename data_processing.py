import numpy as np
import pandas as pd

def read_train_data(dir_path : str = 'data'):
    # reads in data from the dataset
    train_data = np.load('data/images.npy')
    print(train_data.shape)
    train_labels = np.load('data/labels.npy')
    print(train_labels.shape)
    # separates the data into training data/labels and returns them
    return train_data, train_labels

def read_test_data(dir_path : str = 'data'):
    # reads in data from the dataset
    test_data = np.load('data/test_images.npy')
    print(test_data.shape)
    test_labels = np.load('data/test_labels.npy')
    print(test_labels.shape)
    # separates the data into testing data/labels and returns them
    return test_data, test_labels

def read_batch_train(batch_size : int = 32):
    train_x, train_y = read_train_data()
    print(train_x.shape, train_y.shape)
    indices = np.arange(len(train_y))
    np.random.shuffle(indices)
    for batch_i in range(len(train_y)//batch_size):
        idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
        yield np.moveaxis(train_x[idx],-1,1).astype(np.float32)/255, train_y[idx]
    # yields a generator for batches of data

# train_data, train_label, test_data, test_label = read_data()
# print(train_data[0])