import numpy as np
from img_proc_helper import normalize

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

def read_batch(batch_size : int = 32, train : bool = True):
    if train:
        total_x, total_y = read_train_data()
    else:
        total_x, total_y = read_test_data()
    indices = np.arange(len(total_y))
    np.random.shuffle(indices)
    for batch_i in range(len(total_y)//batch_size):
        idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
        yield normalize(total_x[idx]), total_y[idx]
    # yields a generator for batches of data

# train_data, train_label, test_data, test_label = read_data()
# print(train_data[0])