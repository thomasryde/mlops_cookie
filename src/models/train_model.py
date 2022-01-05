import sys
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from torch import load, save, nn, reshape
from torch.optim import Adam
from torch.utils.data import DataLoader
import hydra

from model import MyAwesomeModel, MyAwesomeModel_2

sys.path.append('src/data')
from make_dataset import CustomDataset

log = logging.getLogger(__name__)
@hydra.main(config_path="config",config_name="config_train.yaml")


def train(cfg):

    org_cwd = hydra.utils.get_original_cwd()
    hparams = cfg.hyperparameters
    directory = cfg.directories

    #Load training data
    data_processed = load(org_cwd + directory.preprocessed_data)
    data = DataLoader(CustomDataset(data_processed["X_train"], data_processed["y_train"]),batch_size = hparams.batch_size, shuffle=True)

    #Load Model
    model = MyAwesomeModel()

    #Optimization Settings
    optimizer = Adam(model.parameters(), lr=hparams.learning_rate)
    criterion = nn.NLLLoss()

    #Actual Training Part
    running_loss = np.zeros((hparams.epochs,))
    log.info("Training day and night")
    for epoch in range(hparams.epochs):
        for images, labels in data:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss[epoch] += loss.item()
        log.info('Epoch: %d - loss=%.3f' % (epoch, running_loss[epoch]))

    save(model.state_dict(), org_cwd + directory.model)
    save(model.state_dict(), f"{os.getcwd()}/model.pth")

    #Make plot of learning curve
    plt.plot(range(hparams.epochs), running_loss) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(org_cwd + '/reports/figures/Training_Loss_Curve.png')

 
if __name__ == '__main__':
    train()

