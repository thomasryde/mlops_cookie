# mlops_cookie

Make sure to be in the root directory.

Training commands with and without epochs and learning rate:
python "src/models/train_model.py"
Training Command with options: python "src/models/train_model.py" -epochs=20 -lr=0.005

Evaluate commands on test (default) or train:
python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth"
python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth" -train=1