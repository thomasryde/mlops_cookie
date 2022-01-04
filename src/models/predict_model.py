import argparse
import sys

from torch import load
from torch.utils.data import DataLoader
from model import MyAwesomeModel, val
import sys
sys.path.append('src/data')
from make_dataset import CustomDataset

# Current Directory: root
# Evaluate Command: python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth"
# Evaluate Command with options: python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth" -train=1

class Evaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(description="Script for evaluating")

        parser.add_argument('-model_weights',required=True)
        parser.add_argument('-data', required=True)
        parser.add_argument('-train', default=0,type=int)

        args = parser.parse_args()
        
        weights = load(args.model_weights)
        model = MyAwesomeModel()
        model.load_state_dict(weights)

        #Load test data
        data_processed = load(args.data)
        if args.train == 0:
            evaluate_set = DataLoader(CustomDataset(data_processed["X_test"], data_processed["y_test"]), batch_size=64, shuffle=True)
        elif args.train == 1:
            evaluate_set = DataLoader(CustomDataset(data_processed["X_train"], data_processed["y_train"]), batch_size=64, shuffle=True)
        else:
            print("An error in the train argument occured")

        #Evaluate
        accuracy = val(model, evaluate_set)
        print(f'Accuracy: {accuracy*100}%')

if __name__ == '__main__':
    Evaluate()