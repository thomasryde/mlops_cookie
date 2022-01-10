from argparse import ArgumentParser
from torch import nn, optim, reshape
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.nn import *
import sys

class MyAwesomeModel(nn.Module):
    #Fully Connected Model
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        print("Printing shape")
        print(x.shape[1])
        print(x.shape[2])
        print(x.shape[3])
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected input to specific shape')
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

class MyAwesomeModel_2(nn.Module):
    #CNN Model
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=3,                             
            ),
            nn.ReLU(), 
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              
                out_channels=16,            
                kernel_size=5,                             
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )

        self.fc1 = nn.Sequential(         
            nn.Linear(1936,128                          
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x):
        
        x = reshape(x,(-1,1,28,28))

        x = self.conv1(x)
        x = self.conv2(x)

        #Fully Connected
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        
        return x

def val(model, data: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data:
            images = images.view(images.shape[0], -1)
            ps = model(images)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total += images.shape[0]
            correct += torch.sum(equals)
    accuracy = float(correct) / float(total)
    return accuracy