from typing import Dict, Callable
import warnings
import argparse
import datetime
import logging
import pickle
import json
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from tqdm import tqdm

from load_trained_sktime_classifier import SktimeClassifier
from shappy import KernelShapExplainer
from plot import plot_weighted_graph
from lemon import LemonExplainer

logger = logging.getLogger('src')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

warnings.filterwarnings('ignore')


def generate_explanations(model : Callable,
                          explainer : Callable,
                          train : Dict, 
                          test : Dict,
                          test_size : int = 1000):
                
    train_x, train_y, labels = train['x'], train['y'], train['labels']
    test_x, test_y = test['x'], test['y']

    _, sampled_indices = train_test_split(np.arange(test_x.shape[0]), test_size= test_size / test_x.shape[0], random_state=42)

    test_x = test_x[sampled_indices]
    test_y = test_y[sampled_indices]

    logger.info(f"Generating explanations...")
    
    w = []
    
    for i in tqdm(range(test_x.shape[0]), desc='generating lime explanations for {}'.format(model.model_name)):
        wi = explainer.explain(test_x[i])
        w.append([wi, sampled_indices[i]])

    w_name = f"{model.model_name}_{explainer.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    w_path = os.path.join('./explanations', w_name)

    with open(w_path, 'wb') as f:
        pickle.dump(w, f)

    logger.info(f"explanation weights saved to : {w_path}")

    return w


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/sktime_config.json', help='Path to config file')
    parser.add_argument('--model', type=str, default='inception', help='Path to model')
    args = parser.parse_args()
    config_path = args.config
    model = args.model

    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"loaded config from : {config_path}")

    with open(config['trainpath'], 'rb') as f:
        traindata = pickle.load(f)
    logger.info(f"loaded training data from : {config['trainpath']}")

    with open(config['testpath'], 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f"loaded eval data from : {config['testpath']}")

    if model == 'inception':
        model_path = "./models/InceptionTimeClassifier_20241016_150042"
    elif model == 'resnet':
        model_path = "./models/ResNetClassifier_20241016_150613"
    elif model == 'rocket':
        model_path = "./models/RocketClassifier_20241016_152453"

    model = SktimeClassifier(model_path)
    logger.info(f"loaded model from : {model_path}")
    logger.info(f"model name : {model.model_name}")
    
    limeconfig = {
        "perturbation_ratio" : 0.7,
        "adjacency_prob" : 0.9, 
        "num_samples" : 10000,
        "segment_size" : 25,
        "sigma" : 0.1
        }

    lemonexp = LemonExplainer(model_fn=model.predict, 
                              perturbation_ratio=limeconfig['perturbation_ratio'],
                              adjacency_prob=limeconfig['adjacency_prob'],
                              num_samples=limeconfig['num_samples'],
                              segment_size=limeconfig['segment_size'],
                              sigma=limeconfig['sigma'])
    logger.info(f"Initialized {lemonexp.__class__.__name__} explainer")
    
    generate_explanations(model, lemonexp, traindata, evaldata)

    # shapconfig = {"algorithm_idx" : 0}

    # shapexp = KernelShapExplainer(model_fn=model.predict, 
    #                               x_background=traindata['x'],
    #                               algorithm_idx=shapconfig['algorithm_idx'])
    # logger.info(f"Initialized {shapexp.__class__.__name__} explainer")   

    # generate_explanations(model, shapexp, traindata, evaldata)