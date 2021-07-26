import time
import os
import torch
from torch import nn
import numpy as np

class module_3_5(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(module_3_5, self).__init__()
		# always keep input_dim and ouput_dim even!
		self.condense1 = nn.Conv2d(input_dim, input_dim//2, (1,1))
		self.conv3 = nn.Conv2d(input_dim//2, output_dim//2, (3,3), padding='same')
		self.batch_norm1 = nn.BatchNorm2d(output_dim//2)
		self.conv5 = nn.Conv2d(input_dim//2, output_dim//2, (5,5), padding='same')
		self.batch_norm2 = nn.BatchNorm2d(output_dim//2)

	def forward(self, x):
		x = self.condense1(x)
		x3 = self.conv3(x)
		x3 = self.batch_norm1(x3)
		x5 = self.conv5(x)
		x5 = self.batch_norm2(x5)
		return torch.cat([x3,x5],dim=1) # pytorch is channel first!

class SignRecogCNN(nn.Module):
	def __init__(self):
		super(SignRecogCNN, self).__init__()
		# size: 28
		self.module1 = module_3_5(3,128)
		self.module2 = module_3_5(128,128)
		self.pool1 = nn.MaxPool2d((3,3),2)
		# size: 13
		self.module3 = module_3_5(128,128)
		self.module4 = module_3_5(128,128)
		self.pool2 = nn.MaxPool2d((2,2),2)
		# size: 6
		self.module5 = module_3_5(128,128)
		self.module6 = module_3_5(128,128)
		self.pool2 = nn.MaxPool2d((2,2),2)
