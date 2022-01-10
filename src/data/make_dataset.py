# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import normalize
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    X_train, y_train = None, None
    X_test, y_test = None, None

    input_filepath = input_filepath + '/*.npz'
    pre_processed_data_file = output_filepath + '/pre_processed_data.pth'

    for file in glob(input_filepath):
        D = np.load(file)
        X, y = D['images'], D['labels']

        if 'test' in file:
            X_test = X
            y_test = y
        else:
            if X_train is None:
                X_train = X
                y_train = y
            else:
                X_train = np.vstack((X_train, X))
                y_train = np.hstack([y_train,y])

    #Normalization
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)

    pre_processed_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    torch.save(pre_processed_data,pre_processed_data_file)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
hwa = 2

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        im, label = self.X[idx], self.y[idx]

        if self.transform:
            im = self.transform(label)

        return im, label