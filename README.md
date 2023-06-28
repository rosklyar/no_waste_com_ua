# nowaste.com.ua secondary materials classifier project

## Project description

This project is a part of the [nowaste.com.ua](https://nowaste.com.ua) project. It is a classifier for secondary materials.

## How to train models
### Download data
1. Download data from [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). Or use your own dataset with the same structure(classes are separeted by folders).
2. Unpack data to `./data/kaggle-ds` folder for example.
### Training routine
You can train model using next cli commands consecutively:
1. Separate data to train and test sets
```
python .\src\garbage_classifier\cli.py split-data-folder .\data\kaggle-ds .\data\splitted\ 0.2
```
Output: `.\data\splitted\train` and `.\data\splitted\test` folders with train and test data.
2. Train model
Login to wandb using `wandb login` command. Then run training:
```
python .\src\garbage_classifier\cli.py train-and-log-wandb .\config\training.json .\data\splitted\train .\data\splitted\test .\data\model\output  special-project
```
Or you can run training without logging to wandb:
```
python .\src\garbage_classifier\cli.py train-and-save .\config\training.json .\data\splitted\train .\data\splitted\test .\data\model\output
```
Output: model and it's model card inside `.\data\model\output` folder.
3. Upload model to wandb registry
Note! You need to login to wandb using `wandb login` command.
```
python .\src\garbage_classifier\cli.py upload-to-registry wandb-entity-name special-project model-name .\data\model\output .\config\classes.json
```