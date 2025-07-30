import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Function to transform images to pytorch tensors
transform =transforms.Compose([transforms.ToTensor()])

# Download CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root="./data train" , train=True, download=True, transform = transform)

trainloader= torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(3,16,3 ,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2048,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
  

        return x
  
        

  

