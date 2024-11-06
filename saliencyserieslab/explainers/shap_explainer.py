from typing import Callable
import datetime
import pickle
import json
import os

import numpy as np
import shap

from saliencyserieslab.load_sktime_classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph


class KernelShapExplainer:
    def __init__(
            self, 
            model_fn : Callable, 
            x_background : np.ndarray, 
            algorithm : str="linear", 
            n_background : int=100
            ):

        x_background = shap.sample(x_background, n_background)
        self.explainer = shap.KernelExplainer(model_fn, x_background, algorithm=algorithm)


    def explain_instance(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


class GradientShapExplainer:
    def __init__(
            self, 
            model_fn : Callable, 
            x_background : np.ndarray, 
            algorithm : str="linear", 
            n_background : int=100
            ):

        x_background = shap.sample(x_background, n_background)
        self.explainer = shap.GradientExplainer(model_fn, x_background, algorithm=algorithm)
    

    def explain_instance(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


if __name__ == "__main__":
    
    model = SktimeClassifier()

    model.load_pretrained_model("./models/inception_1")

    datapath = "./data/insectsound/insectsound_test_n10.pkl"
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    print("loaded {} instances from : {}".format(data['x'].shape[0], datapath))

    explainer = GradientShapExplainer(
        model_fn=model.predict, 
        x_background=data['x'],
        algorithm="linear",
        n_background=100,
        )
    
    explainer = KernelShapExplainer(
        model_fn=model.predict, 
        x_background=data['x'],
        algorithm="linear",
        n_background=100,
        )

    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    sample = data['x'][0]
    
    print("explaining sample : {}".format(sample.shape))

    w = explainer.explain_instance(sample)

    w = w.reshape(-1)
    sample = sample.reshape(-1)

    plot_weighted_graph(sample, w, f"./plots/lime_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")