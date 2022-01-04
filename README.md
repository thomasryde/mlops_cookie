Make sure to be in the root directory.

Training commands with and without epochs and learning rate:

```{python}
python "src/models/train_model.py"
python "src/models/train_model.py" -epochs=20 -lr=0.005
```

Evaluate commands on test (default) or train:

```{python}
python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth"
python "src/models/predict_model.py" -data="data/processed/pre_processed_data.pth" -model_weights="src/models/model.pth" -train=1
```

Generate procesed data:

```{python}
python "src/data/make_dataset.py" "data/raw" "data/processed"
```