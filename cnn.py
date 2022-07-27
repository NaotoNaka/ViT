import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,15,4)
        self.conv2 = nn.Conv2d(15,45,4)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(1125,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,i):
        x=self.conv1(i)
        x=self.pool(self.relu(x))
        x=self.conv2(x)
        x=self.pool(self.relu(x))
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.fc2(self.relu(x))
        return x