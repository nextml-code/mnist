# mnist toy problem
Intended to be used for trying new ideas for workflow and modelling

## Setup

    guild init --skip-tensorflow

## Help

    guild help

## Results
Without searching for better hyperparameters

label      | ~val_categorical_accuracy
-----------|--------------------------
supervised | 56%
mixup      | 60%
semi       | 57% (maybe better with search)
mixmatch   | 67% (only train-predicted)
mixmatch   | 50%
