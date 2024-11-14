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
from saliencyserieslab.load_data import UcrDataset


def train_classifier(
        model, 
        save_model : bool,
        train_x : np.ndarray,
        train_y : np.ndarray, 
        eval_x : np.ndarray,
        eval_y : np.ndarray,
        model_save_path : str,
        ):
    
    if os.path.exists(model_save_path):
        raise RuntimeError("model already exists at {}".format(model_save_path))

    print("starting training...")
    start = time.time()
    model.fit(train_x, train_y)
    end = time.time()   
    print("finished training")
    
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
    parser.add_argument('--savedir', type=str, default='./models', help='path to save model')
    parser.add_argument('--id', type=str, default="1010101", help='model id')
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL_NAME = args.model
    MODEL_ID = args.id
    savedir = args.savedir
    save = args.save if args.save else False
    
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    ucr = UcrDataset(
        name=DATASET,
        float_dtype=16,
        scale=True,
        n_dims=16,
    )

    train_x, train_y = ucr.load_split("train")
    test_x, test_y = ucr.load_split("test")

    unique_classes = np.unique(train_y).tolist()

    print("dataset : {}".format(DATASET))
    print("train : {}, {}".format(train_x.shape, train_y.shape))
    print("test : {}, {}".format(test_x.shape, test_x.shape))
    print("number of classes : {}".format(len(unique_classes)))
    time.sleep(3)

    model = SktimeClassifier()
    model.load_model(MODEL_NAME)

    print("loaded model : {}".format(model.name))

    print("Params : ")
    pprint(model.model.__dict__, sort_dicts=False)
    print()

    model_save_path = os.path.join(savedir, f"{MODEL_NAME}_{DATASET}_{MODEL_ID}")

    train_classifier(
        model=model,
        train_x=train_x,
        train_y=train_y,
        eval_x=test_x,
        eval_y=test_y,
        save_model=save,
        model_save_path=model_save_path,
    )