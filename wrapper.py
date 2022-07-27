from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from vit import ViT
from cnn import CNN

from config import *

def download():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(imageWH),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainData = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=True,
        download=True,
        transform=transform
    )
    testData = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=False,
        download=True,
        transform=transform
    )
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        batch_size=imageBatch,
        shuffle=shuffle,
        num_workers=num_workers
    )
    testLoader = torch.utils.data.DataLoader(
        testData,
        batch_size=imageBatch,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return trainLoader, testLoader, len(trainData), len(testData)

class Wrapper():
    def __init__(self,modelIndex):
        self.trainLoader, self.testLoader, self.trainLen, self.testLen = download()
        if(modelIndex == 0):
            self.model = ViT().to(device)
        else:
            self.model = CNN().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr)
    def train(self,epoch = 10):
        self.model.train()
        for i in range(epoch):
            print('-'*200)
            print(f"Epoch {i+1}")
            trainLoss = 0
            for x, y in tqdm(self.trainLoader,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out,y)
                loss.backward()
                self.optimizer.step()
                trainLoss += loss.item()
            print(f"\t\t\t Training Loss:{trainLoss/self.trainLen}")
            self.test()
    def test(self):
        self.model.eval()
        correct = 0
        log=[]
        for x, y in tqdm(self.testLoader,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                out = self.model(x)
            out = torch.argmax(out,axis=1)
            correct += torch.sum(torch.eq(out,y)).item()
            log.append(torch.sum(torch.eq(out,y)).item())
        print(f"Test Accuracy \t\t {correct/self.testLen}")

if(__name__ == "__main__"):
    vitWrapper = Wrapper(0)
    vitWrapper.train()
    vitWrapper.test()