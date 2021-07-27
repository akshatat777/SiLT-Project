import torch
from torch import nn

class module_3_5(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(module_3_5, self).__init__()
		# always keep input_dim and ouput_dim even!
		# 512 --> 256
		self.condense1 = nn.Conv2d(input_dim, input_dim//2, (1,1))
		self.conv3 = nn.Conv2d(input_dim//2, output_dim//2, (3,3), padding='same')
		# 3 by 3 convnet
		self.batch_norm1 = nn.BatchNorm2d(output_dim//2)
		# batch normalization
		self.conv5 = nn.Conv2d(input_dim//2, output_dim//2, (5,5), padding='same')
		# 5 by 5 convnet
		self.batch_norm2 = nn.BatchNorm2d(output_dim//2)
		self.relu = nn.ReLU()

	def forward(self, x):
		# forward pass
		x = self.relu(self.condense1(x))
		x3 = self.conv3(x)
		x3 = self.relu(self.batch_norm1(x3))
		x5 = self.conv5(x)
		x5 = self.relu(self.batch_norm2(x5))
		return torch.cat([x3,x5],dim=1) # pytorch is channel first!

class SignRecogCNN(nn.Module):
	def __init__(self):
		super(SignRecogCNN, self).__init__()
		# size: 100
		self.conv1 = nn.Conv2d(3,128,(3,3),padding='same')
		self.module1 = module_3_5(128,128)
		self.module2 = module_3_5(128,128)
		self.pool1 = nn.MaxPool2d((3,3),2)
		# MaxPool2d "slides the windows"
		# size: 49
		self.module3 = module_3_5(128,128)
		self.module4 = module_3_5(128,128)
		self.pool2 = nn.MaxPool2d((2,2),2)
		# size: 24
		self.module5 = module_3_5(128,128)
		self.module6 = module_3_5(128,128)
		self.pool3 = nn.MaxPool2d((2,2),2)
		# size: 11
		self.module7 = module_3_5(128,128)
		self.module8 = module_3_5(128,128)
		self.pool4 = nn.MaxPool2d((2,2),2)
		# size: 5
		self.module9 = module_3_5(128,128)
		self.module10 = module_3_5(128,128)
		# global average pooling
		self.dense1 = nn.Linear(128, 512)
		self.batch_norm1 = nn.BatchNorm1d(512)
		self.dense2 = nn.Linear(512, 26)
		self.relu = nn.ReLU()

	def forward(self, x):
		# forward pass
		x = self.relu(self.conv1(x))
		# just a bunch of arithmetic
		x1 = self.module1(x)
		x2 = self.module2(x1)
		x = self.pool1(x1+x2)
		x3 = self.module3(x)
		x4 = self.module4(x3)
		x = self.pool2(x3+x4)
		x5 = self.module5(x)
		x6 = self.module6(x5)
		x = self.pool3(x5+x6)
		x7 = self.module7(x)
		x8 = self.module8(x7)
		x = self.pool4(x7+x8)
		x9 = self.module9(x)
		x10 = self.module10(x9)
		x = x9+x10
		# shape (N,128,4,4)
		x = torch.flatten(x, start_dim=2, end_dim=3)
		x = torch.mean(x, dim=-1)
		x = self.batch_norm1(self.relu(self.dense1(x)))
		x = self.dense2(x)
		return x

	