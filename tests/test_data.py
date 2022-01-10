from torch import load, Size
from torch.utils.data import DataLoader
import sys
sys.path.append('src/data')
from make_dataset import CustomDataset
N_train = 25000
N_test = 5000

import os.path
import pytest
@pytest.mark.skipif(not os.path.exists("data/processed/pre_processed_data.pth"), reason="Data files not found")
def test_something_about_data():
    dataset = load("data/processed/pre_processed_data.pth")
    data_train = DataLoader(CustomDataset(dataset["X_train"], dataset["y_train"]), batch_size=1, shuffle=True)
    assert len(data_train) == N_train, "Train set did not have the correct number of samples"
    data_test = DataLoader(CustomDataset(dataset["X_test"], dataset["y_test"]), batch_size=1, shuffle=True)
    assert len(data_test) == N_test, "Test set did not have the correct number of samples"
    dataiter = iter(data_train)
    images, labels = dataiter.next()
    assert images.shape == Size([1, 28, 28]), "Wrong image shape"
    assert all(i in dataset["y_train"] for i in range(10)), "Not all labels is represented 0..9"