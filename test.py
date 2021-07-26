import torch

x = torch.zeros((3,4,5,6))
print(torch.flatten(x,start_dim=2,end_dim=3).shape)