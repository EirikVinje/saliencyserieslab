from typing import Callable
import datetime
import pickle
import json
import os

import numpy as np
import shap

from saliencyserieslab.plotting import plot_simple_weighted
from saliencyserieslab.classifier import SktimeClassifier
from aeon.datasets import load_classification


class KernelShapExplainer:
    def __init__(
            self, 
            model, 
            x_background : np.ndarray, 
            algorithm : str="linear", 
            n_background : int=None
            ):

        if n_background is not None:
            x_background = shap.sample(x_background, n_background)    

        self.explainer = shap.KernelExplainer(model.predict, x_background, algorithm=algorithm)


    def explain_instance(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


class GradientShapExplainer:
    def __init__(
            self, 
            model, 
            x_background : np.ndarray, 
            algorithm : str="linear", 
            n_background : int=None,
            ):

        if n_background is not None:
            x_background = shap.sample(x_background, n_background)

        self.explainer = shap.GradientExplainer(model.predict, x_background, algorithm=algorithm)
    

    def explain_instance(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


if __name__ == "__main__":
    
    np.random.seed(42)

    modelpath = "./models/rocket_ECG200_1"
    dataset = modelpath.split("/")[-1].split("_")[1]

    model = SktimeClassifier()
    model.load_pretrained_model(modelpath)

    test = load_classification(dataset, split="test")

    unique_classes = np.unique(test[1]).tolist()
    unique_classes = [int(c) for c in unique_classes]
    unique_classes.sort()
 
    test = {
        "x" : test[0].squeeze(),
        "y" : np.array([unique_classes.index(int(c)) for c in test[1]]),
    }

    method = "kernel"

    if method == "kernel":

        explainer = KernelShapExplainer(
            model=model,
            x_background=shap.sample(test["x"], 1),
        )

    elif method == "grad":

        explainer = GradientShapExplainer(
            model=model,
            x_background=shap.sample(test["x"], 1)
        )

    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    sample = test["x"][np.random.randint(0, test["x"].shape[0])]

    w = explainer.explain_instance(sample)

    sample = sample.reshape(-1)
    w = w.reshape(-1)

    plot_simple_weighted(
        sample, 
        w, 
        f"./plots/{method}shap_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png",
        model.model.__class__.__name__,
        explainer.__class__.__name__,
        dataset
        )
