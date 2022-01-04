import argparse
import sys

from torch import load, save
from model import MyAwesomeModel
from torch.optim import Adam
from torch import nn
from argparse import ArgumentParser

from torch.utils.data import DataLoader
import sys
sys.path.append('src/data')
from make_dataset import CustomDataset

class Train(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):

        #Parser Information
        parser = argparse.ArgumentParser(
            description="Script for training",
        )
        parser.add_argument('-epochs', default=10, type=int)
        parser.add_argument('-lr', default=0.001, type=float)
        args = parser.parse_args()
        
        #Load training data
        data_processed = load("data/processed/pre_processed_data.pth")
        data = DataLoader(CustomDataset(data_processed["X_train"], data_processed["y_train"]), batch_size=64, shuffle=True)

        #Load Model
        model = MyAwesomeModel()

        #Optimization Settings
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()

        #Actual Training Part
        print("Training day and night")
        for epoch in range(args.epochs):
            running_loss = 0
            model.train()
            for images, labels in data:
                optimizer.zero_grad()
                images = images.view(images.shape[0], -1)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('Epoch: %d - loss=%.3f' % (epoch, running_loss))

        save(model.state_dict(), 'model.pth')
 
if __name__ == '__main__':
    Train()

