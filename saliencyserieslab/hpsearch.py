import argparse
import logging
import pickle
import gc
import os

from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score
import numpy as np
import optuna

from saliencyserieslab.amee_classifier import SktimeClassifier
from saliencyserieslab.amee_train_classifier import train
from generate_config import generate_hp_config


def objective(trail):
    
    hpconfig = generate_hp_config(MODEL_NAME, trail)

    model = SktimeClassifier(hpconfig)
    
    model.fit(train['x'], train['y'])

    accuracy = model.evaluate(test['x'], test['y'])

    return accuracy
    
    
if  __name__ == "__main__":
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ECG5000', help='Path to train data')
    parser.add_argument('--model', type=str, default="inception", help='sktime model [inception, rocket, resnet]')
    parser.add_argument('--trial', type=int, default=10, help='number of trials')
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL_NAME = args.model
    
    train = load_classification(DATASET, split="train")
    test = load_classification(DATASET, split="test")

    unique_classes = np.unique(train[1]).tolist()
    unique_classes = [int(c) for c in unique_classes]
    unique_classes.sort()
 
    train = {
        "x" : train[0].squeeze(),
        "y" : np.array([unique_classes.index(int(c)) for c in train[1]]),
    }

    test = {
        "x" : test[0].squeeze(),
        "y" : np.array([unique_classes.index(int(c)) for c in test[1]]),
    }

    scaler = StandardScaler()
    train['x'] = scaler.fit_transform(train['x'])
    test['x'] = scaler.transform(test['x'])

    model = SktimeClassifier()
    model.load_model(MODEL_NAME)

    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///optuna.db", 
        study_name=f"{MODEL_NAME}_{DATASET}_study")
    
    study.optimize(objective, n_trials=args.trial)