import argparse
import sys
import hydra
import logging
from torch import load
from torch.utils.data import DataLoader
from model import MyAwesomeModel, MyAwesomeModel_2, val

sys.path.append('src/data')
from make_dataset import CustomDataset

log = logging.getLogger(__name__)
@hydra.main(config_path="config",config_name="config_test.yaml")


def evaluate(cfg):

    org_cwd = hydra.utils.get_original_cwd()
    hparams = cfg.hyperparameters
    directory = cfg.directories
    
    weights = load(org_cwd + directory.model)
    model = MyAwesomeModel()
    model.load_state_dict(weights)

    #Load test data
    data_processed = load(org_cwd + directory.preprocessed_data)
    if hparams.train == 0:
        evaluate_set = DataLoader(CustomDataset(data_processed["X_test"], data_processed["y_test"]), batch_size=128, shuffle=True)
    elif hparams.train == 1:
        evaluate_set = DataLoader(CustomDataset(data_processed["X_train"], data_processed["y_train"]), batch_size=128, shuffle=True)
    else:
        log.info("An error in the train argument occured")

    #Evaluate
    accuracy = val(model, evaluate_set)
    log.info(f'Accuracy: {accuracy*100}%')

if __name__ == '__main__':
    evaluate()