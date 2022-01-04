from argparse import ArgumentParser
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import sys

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
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