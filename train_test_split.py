import numpy as np

joints = np.load('data/n_joints.npy')
labels = np.load('data/n_labels.npy')

test_split = 0.2

train_len = int(len(joints)*(1-test_split))
train_joints = joints[:train_len]
test_joints = joints[train_len:]
train_labels = labels[:train_len]
test_labels = labels[train_len:]

np.save('data/n_joints_train',train_joints.astype(np.float32))
np.save('data/n_joints_test',test_joints.astype(np.float32))
np.save('data/n_labels_train',train_labels.astype(np.int64))
np.save('data/n_labels_test',test_labels.astype(np.int64))