Make sure to be in the root directory.

Training commands with and without custom epochs and learning rate:

```{python}
python "src/models/train_model.py"
python "src/models/train_model.py" hyperparameters.epochs=20 hyperparameters.learning_rate=0.005
```

Evaluate commands on test (default) or train:

```{python}
python "src/models/predict_model.py" hyperparameters.train=1
```

Generate procesed data:

```{python}
python "src/data/make_dataset.py" "data/raw" "data/processed"
```