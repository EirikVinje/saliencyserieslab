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

from load_sktime_classifier import SktimeClassifier

logger = logging.getLogger('saliencyserieslab')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def train(model, traindata, evaldata):

    logger.info("Running {}".format(model.__class__.__name__))

    model_save_path = f'./models/{MODEL_NAME}_{MODEL_ID}'
    
    if os.path.exists(model_save_path):
        logger.info(f"model already exists at {model_save_path}")
        logger.info("delete file or change model id")
        return

    train_x, train_y, labels = traindata['x'], traindata['y'], traindata['labels']

    logger.info(f"train size : {train_x.shape}")
    
    model.fit(train_x, train_y)

    eval_x, eval_y = evaldata['x'], evaldata['y']

    logger.info(f"test size : {eval_x.shape}")

    predictions = model.predict(eval_x)
    accuracy = accuracy_score(eval_y, predictions)
    print()
    print(f'Accuracy: {accuracy}')
    print()
    
    report = classification_report(eval_y, predictions, target_names=labels, output_dict=True)

    for label, item in list(report.items())[:len(labels)]:
        print(f'{label} : precision : {item["precision"]}')
    print()

    mlflow_sktime.save_model(model, model_save_path)
    logger.info(f'Saved model to : {model_save_path}')    


if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', type=str, default='./data/insectsound/insectsound_train.pkl', help='Path to train data')
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--model', type=str, default="inceptiontime", help='sktime model [inception, rocket, resnet]')
    parser.add_argument('--id', type=str, help='model id')
    args = parser.parse_args()

    trainpath = args.trainpath
    testpath = args.testpath
    MODEL_NAME = args.model
    MODEL_ID = args.id

    logger.info("Running {}".format(__file__))

    with open(trainpath, 'rb') as f:
        traindata = pickle.load(f)
    logger.info(f'Loaded train data from {trainpath}')

    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f'Loaded eval data from {testpath}')

    model = SktimeClassifier()
    model.load_model(MODEL_NAME)
    
    train(model, traindata, evaldata)



