from typing import Callable
import argparse
import pickle
import json

import numpy as np
import shap
# from timeshap.explainer import local_report


from windowshap import SlidingWindowSHAP, StationaryWindowSHAP, DynamicWindowSHAP
from load_trained_sktime_classifier import SktimeClassifier
from plot import plot_weighted_graph


class ShapRegularExplainer:

    def __init__(self, model_fn : Callable, x : np.ndarray):
        self.explainer = shap.Explainer(model_fn, x)
        
    def explain(self, x : np.ndarray, max_evals : int):

        shap_values = self.explainer(x, max_evals=max_evals)
        w = shap_values.values[0]
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


class TimeShapExplainer:
    def __init__(self):
        pass


class KernelShapExplainer:
    def __init__(self, model_fn : Callable, x_background : np.ndarray, algorithm_idx : int=0):

        x_background = shap.sample(x_background, 50)
        algorithms = ["auto", "permutation", "partition", "tree", "linear"]
        self.explainer = shap.KernelExplainer(model_fn, x_background, algorithm=algorithms[algorithm_idx])

    def explain(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/sktime_config.json', help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(config['trainpath'], 'rb') as f:
        traindata = pickle.load(f)

    with open(config['testpath'], 'rb') as f:
        evaldata = pickle.load(f)

    model_path = "./models/InceptionTimeClassifier_20241016_150042"

    model = SktimeClassifier(model_path)

    idx = 0
    test_y = evaldata['y'][0]
    test_x = evaldata['x'][0]
    
    train_x = traindata['x'][np.where(traindata['y'] == test_y)]
    train_x = shap.sample(train_x, 50)
    
    explainer = shap.KernelExplainer(model.predict, train_x)
    shap_values = explainer.shap_values(test_x, gc_collect=True)

    plot_weighted_graph(test_x, shap_values, 'kernelshap')