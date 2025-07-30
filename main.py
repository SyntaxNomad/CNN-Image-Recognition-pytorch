import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


transform =transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root="./data train" , train=True, download=True, transform = transform)

trainloader= torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()