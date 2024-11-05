import multiprocessing
import argparse
import datetime
import logging
import pickle
import zipfile
import json
import os

from sklearn.metrics import accuracy_score, classification_report
from sktime.utils import mlflow_sktime
import numpy as np

from load_sktime_classifier import SktimeClassifier


def train(model, traindata, evaldata, save_model : bool = False):

    model_save_path = f'./models/{MODEL_NAME}_{MODEL_ID}'
    
    if os.path.exists(model_save_path):
        raise RuntimeError("model already exists at {}".format(model_save_path))

    train_x, train_y, labels = traindata['x'], traindata['y'], traindata['labels']

    model.fit(train_x, train_y)

    eval_x, eval_y = evaldata['x'], evaldata['y']

    predictions = model.predict(eval_x)
    accuracy = accuracy_score(eval_y, predictions)
    
    print()
    print(f'Accuracy: {accuracy}')
    print()
    
    # report = classification_report(eval_y, predictions, target_names=labels, output_dict=True)

    # for label, item in list(report.items())[:len(labels)]:
    #     print(f'{label} : precision : {item["precision"]}')
    # print()

    if save_model:
        mlflow_sktime.save_model(model, model_save_path)
        
        metaconfig = {
            "accuracy" : accuracy,
            "n_classes" : np.unique(train_y).shape[0],
            "modelconfig" : model.config,
        }

        with open(f'{model_save_path}/metaconfig.json', 'w') as f:
            json.dump(metaconfig, f)

    return accuracy

if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', type=str, default='./data/insectsound/insectsound_train_n10.pkl', help='Path to train data')
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test_n10.pkl', help='Path to eval data')
    parser.add_argument('--model', type=str, default="inception", help='sktime model [inception, rocket, resnet, mrseql]')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False, help='save model')
    parser.add_argument('--id', type=str, help='model id')
    args = parser.parse_args()

    trainpath = args.trainpath
    testpath = args.testpath
    MODEL_NAME = args.model
    MODEL_ID = args.id
    save = args.save if args.save else False

    with open(trainpath, 'rb') as f:
        traindata = pickle.load(f)

    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)

    print("train size : {}".format(traindata['x'].shape[0]))
    print("eval size : {}".format(evaldata['x'].shape[0]))

    with open("./modelconfigs/config.json", 'r') as f:
        modelconfig = json.load(f)

    model = SktimeClassifier(modelconfig)
    model.load_model(MODEL_NAME)
    
    train(model, traindata, evaldata, save)



