import argparse
import logging
import pickle
import json
import gc
import os

from tqdm import tqdm
import numpy as np

from load_trained_sktime_classifier import SktimeClassifier
from generate_perturbed_data import PerturbedDataGenerator

logger = logging.getLogger('src')


def compute_eauc(model, explanation_weights : np.ndarray, test_x : np.ndarray, test_y : np.ndarray):

    all_accuracy = []
    k = np.arange(0.0, 1.01, 0.1)
    
    generator = PerturbedDataGenerator(explanation_weights, test_x, test_y)
    
    for ki in tqdm(k):

        perturbed_x, y = generator.generate(ki)
        acc = model.evaluate(perturbed_x, y)
        all_accuracy.append(acc)

    eauc = np.trapz(x=k, y=all_accuracy)

    print(f"all accuracies : {all_accuracy}")
    print(f"EAUC : {eauc}")

    return eauc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/sktime_config.json', help='Path to config file')
    parser.add_argument('--model', type=str, default='inception')
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

    explanation_weights_path = "./explanations/InceptionTimeClassifier_LemonExplainer_20241021_101934.pkl"

    with open(explanation_weights_path, 'rb') as f:
        w = pickle.load(f)
    logger.info(f"loaded explanation weights from method : {explanation_weights_path.split('_')[1]}")

    compute_eauc(model, w, evaldata['x'], evaldata['y'])
