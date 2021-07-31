import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

class SignRecogCNN(nn.Module):
	def __init__(self):
		super(SignRecogCNN, self).__init__()
		mobilenet = mobilenet_v3_large(pretrained=True)
		extract_layers = list(mobilenet.children())[0][:11]
		self.feature_extract = nn.Sequential(*extract_layers)
		for param in self.feature_extract.parameters():
			param.requires_grad = False
		# 14*14*80
		self.dropout0 = nn.Dropout(0.3)
		self.conv1 = nn.Conv2d(80,64,(3,3),padding='same')
		self.batchnorm1 = nn.BatchNorm2d(64)
		self.dropout1 = nn.Dropout(0.3)
		self.conv2 = nn.Conv2d(64,64,(3,3),padding='same')
		self.batchnorm2 = nn.BatchNorm2d(64)
		self.dropout2 = nn.Dropout(0.3)
		self.pool1 = nn.MaxPool2d((3,3),stride=2)
		# 6*6*64
		self.conv3 = nn.Conv2d(64,64,(3,3))
		self.batchnorm3 = nn.BatchNorm2d(64)
		self.dropout3 = nn.Dropout(0.3)
		# 4*4*64
		self.conv4 = nn.Conv2d(64,96,(2,2),padding='same')
		self.batchnorm4 = nn.BatchNorm2d(96)
		self.dropout4 = nn.Dropout(0.3)

		self.dense1 = nn.Linear(96,96)
		self.batchnorm6 = nn.BatchNorm1d(96)
		self.dropout5 = nn.Dropout(0.3)
		self.dense2 = nn.Linear(96,26)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.dropout0(self.feature_extract(x))
		x = self.dropout1(self.batchnorm1(self.relu(self.conv1(x))))
		x = self.dropout2(self.batchnorm2(self.relu(self.conv2(x))))
		x = self.pool1(x)
		x = self.dropout3(self.batchnorm3(self.relu(self.conv3(x))))
		x = self.dropout4(self.batchnorm4(self.relu(self.conv4(x))))
		x = torch.max(torch.flatten(x, start_dim=2, end_dim=3),dim=2).values
		x = self.dropout5(self.batchnorm6(self.relu(self.dense1(x))))
		return self.dense2(x)
