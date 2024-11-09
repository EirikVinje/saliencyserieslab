from pprint import pprint
import multiprocessing
import argparse
import datetime
import logging
import pickle
import zipfile
import json
import time
import csv
import os

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification
from sktime.utils import mlflow_sktime
import numpy as np

from saliencyserieslab.classifier import SktimeClassifier
from generate_config import generate_config


def train_classifier(model, traindata, evaldata, save_model : bool = False):

    model_save_path = f'./models/{MODEL_NAME}_{DATASET}_{MODEL_ID}'
    
    if os.path.exists(model_save_path):
        raise RuntimeError("model already exists at {}".format(model_save_path))

    train_x, train_y = traindata['x'], traindata['y']

    print("starting training...")
    start = time.time()
    model.fit(train_x, train_y)
    end = time.time()   
    print("finished training")

    eval_x, eval_y = evaldata['x'], evaldata['y']
    
    print("evaluating model...")
    predictions = model.predict(eval_x, verbose=True)
    print("finished evaluation")
    
    accuracy = accuracy_score(eval_y, predictions)
    
    print()
    print(f'Accuracy: {accuracy}')
    print()
    
    report = classification_report(eval_y, predictions, output_dict=True)

    for label, item in list(report.items())[:np.unique(train_y).shape[0]]:
        print(f'{label} : precision : {item["precision"]}')
    print()

    if save_model:
        mlflow_sktime.save_model(model.model, model_save_path)
        
        training_report = {
            "accuracy" : accuracy,
            "precisions" : list(report.items())[np.unique(train_y).shape[0]:],
            "training_time" : "{} minutes".format((end - start) / 60),
        }

        with open(f'{model_save_path}/training_report.json', 'w') as f:
            json.dump(training_report, f)

        print("model and results saved to {}".format(model_save_path))
        time.sleep(3)

    
    if not os.path.isfile("./training_report.csv"):

        with open("./training_report.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "model", "accuracy", "training_time", "datetime"])

    with open("./training_report.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([DATASET, MODEL_NAME, accuracy, training_report["training_time"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    print("training report saved in {}".format("./training_report.csv"))


if __name__ == '__main__':
    
    """
    Datasets:
    - ECG200
    - ECG5000
    - SwedishLeaf
    - Epilepsy2

    """

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Plane", help='dataset to load from ucr archive')
    parser.add_argument('--model', type=str, default="inception", help='sktime model [inception, rocket, resnet, mrseql]')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False, help='save model')
    parser.add_argument('--id', type=str, default="1010101", help='model id')
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL_NAME = args.model
    MODEL_ID = args.id
    save = args.save if args.save else False

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

    print("dataset : {}".format(DATASET))
    print("train : {}, {}".format(train['x'].shape, train['y'].shape))
    print("test : {}, {}".format(test['x'].shape, test['y'].shape))
    print("number of classes : {}".format(len(unique_classes)))
    time.sleep(3)

    model = SktimeClassifier()
    model.load_model(MODEL_NAME)

    print("loaded model : {}".format(model.name))

    print("Params : ")
    pprint(model.model.__dict__, sort_dicts=False)
    print()

    train_classifier(model, train, test, save)