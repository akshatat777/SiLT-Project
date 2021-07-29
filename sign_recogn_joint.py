import torch
from torch import nn

class SignRecogJoint(nn.Module):
    def __init__(self):
        super(SignRecogJoint, self).__init__()
        self.dense1 = nn.Linear(63,128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128,128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(128,128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(128,26)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.dropout1(self.batch_norm1(self.relu(self.dense1(x))))
        x = self.dropout2(self.batch_norm2(self.relu(self.dense2(x))))
        return self.dense4(x)