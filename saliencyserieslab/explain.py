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

# logger = logging.getLogger('src')
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

def explain(explainer : Callable, 
            test_x : np.ndarray,
            savedir : str,
            explainer_name : str,
            model_name : str,
            run_id : str) -> np.ndarray:

    """
    Given a a pair (model, explainer), generates the \n 
    explanation "w" for each xi in test_x.

    :param Callable, The trained model model: 
    :param Callable, The explainer explainer: 
    :param np.ndarray, The test set (x) test_x: 
    :param str, The name of the explainer explainer_name: 
    :param str, The name of the model model_name: 
    :param str, The id run_id:

    :return np.ndarray: 
    """

    w_path = os.path.join(savedir, f"{model_name}_{explainer_name}_{run_id}.pkl")
    
    if os.path.exists(w_path):
        raise RuntimeError("explanation weights already exists at {}".format(w_path))
    
    w = []
    
    for i in tqdm(range(test_x.shape[0]), desc='generating explanations'):
        wi = explainer(test_x[i])
        w.append(wi)

    w = np.vstack(w)

    with open(w_path, 'wb') as f:
        pickle.dump(w, f)

    return w


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--savedir', type=str, help='path to save the explanation')
    parser.add_argument('--explainer', type=str, help='name of the explainer')
    parser.add_argument('--model', type=str, help='name of the model')
    parser.add_argument('--id', type=str, help='model id')
    args = parser.parse_args()
    
    with open(args.testpath, 'rb') as f:
        evaldata = pickle.load(f)
    
    model_path = f"./models/{args.model}_{args.id}"

    model = SktimeClassifier()
    model.load_pretrained_model(model_path)
    
    test_x = evaldata['x']

    explainer = Explainer()
    explainer.load_explainer(explainer=args.explainer, model_fn=model.predict, background_data=test_x)

    explain(explainer, test_x, args.savedir, args.explainer, args.model, args.id)