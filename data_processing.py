import numpy as np
import cv2

def random_rotation(total_x):
    rand_a = np.random.uniform(-1,1)*np.pi/3
    rand_b = np.random.uniform(-1,1)*np.pi/3
    rand_c = np.random.uniform(-1,1)*np.pi/3
    rotA = np.array([[1,0,0],[0,np.cos(rand_a),-np.sin(rand_a)],[0,np.sin(rand_a),np.cos(rand_a)]])
    rotB = np.array([[np.cos(rand_b),0,np.sin(rand_b)],[0,1,0],[-np.sin(rand_b),0,np.cos(rand_b)]])
    rotC = np.array([[np.cos(rand_c),-np.sin(rand_c),0],[np.sin(rand_c),np.cos(rand_c),0],[0,0,1]])
    return total_x @ rotA @ rotB @ rotC

def random_scale(total_x):
    rand_scale = np.random.uniform(0.5,2)
    return total_x * rand_scale

def random_flip(total_x):
    if np.random.randint(0,2) == 0:
        return -total_x
    else:
        return total_x

def data_augmentation(total_x):
    # (N,1,21,3)
    total_x = random_rotation(total_x)
    total_x = random_scale(total_x)
    total_x = random_flip(total_x)
    return total_x

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
            yield normalize_joints(total_x[idx]), total_y[idx]
        else:
            #imgs = [cv2.copyMakeBorder(img, 0,224-img.shape[0],0,224-img.shape[1],cv2.BORDER_CONSTANT,0) for img in total_x[idx]]
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

def resize(image):
    image = cv2.resize(image, (70, 70), interpolation=cv2.INTER_AREA)
    return image

def normalize(imgs):
    return swapaxis(imgs).astype(np.float32)/255

def normalize_joints(total_x):
    total_x = total_x - total_x[:,:,:1,:]
    factor = np.mean(np.linalg.norm(total_x-total_x[:,:,0:1,:],axis=-1,keepdims=True),axis=-2,keepdims=True)
    if (factor != 0).all():
        total_x /= factor
    return total_x

def swapaxis(imgs):
    return np.moveaxis(imgs,-1,1)

#====================================================================================

# j = np.random.random((1,1,21,3))
# print(normalize_joints(j))