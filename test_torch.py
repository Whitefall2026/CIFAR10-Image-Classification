import torch
import torchvision
import matplotlib.pyplot as plt
import sklearn
print(torch.__version__)

print(torchvision.__version__)

print(torch.cuda.is_available())

x = torch.randn(3, 32, 32)
print(x.shape)