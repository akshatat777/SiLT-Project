import numpy as np
import cv2

#DATA PROCESSING
def read_data(data_name : str, label_name : str):
    # reads in data from the dataset
    data = np.load(f'data/{data_name}.npy')
    labels = np.load(f'data/{label_name}.npy')
    # separates the data into training data/labels and returns them
    return data, labels

def read_batch(batch_size : int = 32, train : bool = True, joints : bool = True):
    if joints:
        if train:
            total_x, total_y = read_data('n_joints_train','n_labels_train')
        else:
            total_x, total_y = read_data('n_joints_test','n_labels_test')
    else:
        if train:
            total_x, total_y = read_data('images','labels')
        else:
            total_x, total_y = read_data('test_images','test_labels')
    indices = np.arange(len(total_y))
    np.random.shuffle(indices)
    for batch_i in range(len(total_y)//batch_size):
        idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
        if joints:
            yield total_x[idx], total_y[idx]
        else:
            yield normalize(total_x[idx]), total_y[idx]
    # yields a generator for batches of data

#====================================================================================
# IMAGE PROCESSING

def resize_crop(image):
    max_dim = np.argmax(image.shape[:2])
    scale_percent = 100 / image.shape[max_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent) 
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if max_dim == 0:
        image = cv2.copyMakeBorder(image, 0,0,0,100-image.shape[1],cv2.BORDER_CONSTANT,0)
    else:
        image = cv2.copyMakeBorder(image, 0,100-image.shape[0],0,0,cv2.BORDER_CONSTANT,0)
    return image

def normalize(imgs):
    return swapaxis(imgs).astype(np.float32)/255

def swapaxis(imgs):
    return np.moveaxis(imgs,-1,1)

#====================================================================================