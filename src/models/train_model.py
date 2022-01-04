import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

from torch import load, save, nn, reshape
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import MyAwesomeModel, MyAwesomeModel_2

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
        data = DataLoader(CustomDataset(data_processed["X_train"], data_processed["y_train"]), batch_size=128, shuffle=True)

        #Load Model
        model = MyAwesomeModel()

        #Optimization Settings
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()

        #Actual Training Part
        running_loss = np.zeros((args.epochs,))
        print("Training day and night")
        for epoch in range(args.epochs):
            for images, labels in data:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss[epoch] += loss.item()
            print('Epoch: %d - loss=%.3f' % (epoch, running_loss[epoch]))

        save(model.state_dict(), 'src/models/model.pth')

        #Make plot of learning curve
        plt.plot(range(args.epochs), running_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('reports/figures/Training_Loss_Curve.png')

 
if __name__ == '__main__':
    Train()

