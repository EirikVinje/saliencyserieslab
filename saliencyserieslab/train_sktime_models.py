import multiprocessing
import argparse
import datetime
import logging
import pickle
import zipfile
import json
import os

from sklearn.metrics import accuracy_score, classification_report

from sktime.classification.shapelet_based import ShapeletTransformClassifier

from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.deep_learning import ResNetClassifier
from sktime.classification.deep_learning import TapNetClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import WEASEL

from sktime.utils import mlflow_sktime

logger = logging.getLogger('saliencyserieslab')

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def train(model, traindata, evaldata):

    logger.info("Running {}".format(model.__class__.__name__))

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

    model_name = model.__class__.__name__
    model_save_path = f'./models/{model_name}_{TIMESTAMP}' 

    mlflow_sktime.save_model(model, model_save_path)
    logger.info(f'Saved model to : {model_save_path}')    


if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/sktime_config.json', help='Path to config file')
    parser.add_argument('--model', type=str, default="inceptiontime", help='sktime model [inception, rocket, resnet]')
    args = parser.parse_args()

    config_path = args.config
    model_name = args.model

    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Running {}".format(__file__))

    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f'Loaded config from {config_path}')

    with open(config['trainpath'], 'rb') as f:
        traindata = pickle.load(f)
    logger.info(f'Loaded train data from {config["trainpath"]}')

    with open(config['testpath'], 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f'Loaded eval data from {config["testpath"]}')

    # model_args = {}
    # model_args["rocket"] = {"num_kernels" : 5000}
    # model_args["inceptiontime"] = {"n_epochs" : 30, "verbose" : True, "batch_size" : 64}
    # model_args["resnet"] = {"n_epochs" : 30, "verbose" : True, "batch_size" : 64}

    models = {
    "rocket" : RocketClassifier(num_kernels=1000, use_multivariate="no", n_jobs=multiprocessing.cpu_count()-2),
    "inception" : InceptionTimeClassifier(n_epochs=30, verbose=True, batch_size=128, kernel_size=16),
    "resnet" : ResNetClassifier(n_epochs=30, verbose=True, batch_size=128),
    }

    model = models[model_name]

    train(model, traindata, evaldata)


