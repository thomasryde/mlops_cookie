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

Exercises M1-M4 in folder "M1-M4".

Exercises M5-M8: Git & Cookiecutter can be seen using this repo. DVC: We used google drive but later changed to google cloud. DVC was used in project.

Exercises M9-M10: We used hydra. Dockerfile can be seen in repo. Config file can be seen in "src/models/config".

