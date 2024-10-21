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

from saliencyserieslab.load_sktime_classifier import SktimeClassifier
from plot import plot_weighted_graph
from load_explainer import Explainer

logger = logging.getLogger('src')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

warnings.filterwarnings('ignore')

def generate_explanations(explainer : Callable, test_x : np.ndarray) -> np.ndarray:

    """
    Given a a pair (model, explainer), generates the \n 
    explanation "w" for each xi in test_x.

    :param Callable, The trained model model: 
    :param Callable, The explainer explainer: 
    :param np.ndarray, The test set (x) test_x: 

    :return np.ndarray: 

    """

    w_path = f"./explanations/{MODEL_NAME}_{EXPLAINER_NAME}_{MODEL_ID}.pkl"
    
    if os.path.exists(w_path):
        logger.info(f"explanation weights already exists at {w_path}")
        logger.info("delete file or change id")
        return
    
    logger.info(f"Generating explanations...")
    
    w = []
    
    for i in tqdm(range(test_x.shape[0]), desc='generating explanations'):
        wi = explainer(test_x[i])
        w.append(wi)

    w = np.vstack(w)

    with open(w_path, 'wb') as f:
        pickle.dump(w, f)

    logger.info(f"explanation weights saved to : {w_path}")

    return w


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--config', type=str, default='./modelconfigs/sktime_config.json', help='Path to config file')
    parser.add_argument('--explainer', type=str, default='lime', help='name of the explainer')
    parser.add_argument('--model', type=str, default='inception', help='name of the model')
    parser.add_argument('--id', type=str, help='model id')
    args = parser.parse_args()
    
    EXPLAINER_NAME = args.explainer
    config_path = args.config
    testpath = args.testpath
    MODEL_NAME = args.model
    MODEL_ID = args.id

    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f"loaded eval data from : {testpath}")

    model_path = f"./models/{MODEL_NAME}_{MODEL_ID}"

    model = SktimeClassifier()
    model.load_pretrained_model(model_path)
    
    logger.info(f"loaded model from : {model_path}")
    logger.info(f"model name : {model.model_name}")
    
    test_x = evaldata['x']

    explainer = Explainer()
    explainer.load_explainer(explainer=EXPLAINER_NAME, model_fn=model.predict, background_data=test_x)

    logger.info(f"Initialized {explainer.explainer_name} explainer")
    
    generate_explanations(explainer, test_x)