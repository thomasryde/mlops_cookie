This github repos shows the exercises done in the course 02476 Machine Learning Operations at DTU. Exercises were done by Lucas Sørensen (s174461), Mikkel Kofoed Pedersen (s174485), Nicolai Pleth (s174503) and Thomas Ryde (163955).

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

Exercises M1-M4: In folder "M1-M4".

Exercises M5-M8: Git & Cookiecutter can be seen using this repo. DVC: We used google drive but later changed to google cloud. DVC was used in project.

Exercises M9-M10: We used hydra. Dockerfile can be seen in repo. Config file can be seen in "src/models/config".

Exercises M11-M13: In folder "M11-M13". Using WANDB and profiling to find bugs. Found a bug with extra bug with randn/rand.

Exercises M15-M16: One can see in the git repo that we used CI/CD.

Exercise M17-M18: Hard to give evidence that we did this. This was done on our GCP users. Changed DVC storage to google cloud from google drive.

Exercise M19-21: Look in folder "M19-M21". We have not included the data for git space reasons. We found with large batch size we needed high amount of workers. For batch size 100-200 it seemed to work best with 2 or 4 workers. Quanitization allowed for faster inference of NN. Using 8 bits instead of 32 was about 3.5 times faster for inference. Using JIT compilation also made inference faster. MobileNet was also significantly faster than resnet (I believe roughly 4 times faster).

Exercise M22-M23: M23 was done in the cloud. M22 can be seen in the folder 'M22'. We used ResNet50-2 and the cat image. Egyptian Cat was the prediction.



