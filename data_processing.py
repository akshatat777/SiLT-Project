import numpy as np
import pandas as pd

def read_data(dir_path : str = 'data'):
    train_data = pd.read_csv(dir_path+'/sign_mnist_train.csv').to_numpy()
    test_data = pd.read_csv(dir_path+"/sign_mnist_test.csv").to_numpy()

    train_label = train_data[:,0]
    test_label = test_data[:,0]
    train_data = train_data[:,1:]
    test_data = test_data[:,1:]
    train_data = np.reshape(train_data,(train_data.shape[0],28,28))
    test_data = np.reshape(test_data,(test_data.shape[0],28,28))
    train_data = (train_data - np.mean(train_data)) / np.std(train_data)
    test_data = (test_data - np.mean(test_data)) / np.std(test_data)
    return train_data.astype(np.float32), train_label, test_data.astype(np.float32), test_label

# train_data, train_label, test_data, test_label = read_data()
# print(train_data[0])