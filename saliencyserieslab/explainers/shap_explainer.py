from typing import Callable, List
import datetime
import pickle
import json
import os

import numpy as np
import shap

from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.load_data import UcrDataset

class KernelShapExplainer:
    def __init__(
            self, 
            model, 
            x_background : np.ndarray, 
            algorithm : str="linear", 
            ):

        self.explainer = shap.KernelExplainer(model.predict, x_background, algorithm=algorithm)


    def explain_instance(self, x : np.ndarray, y : np.ndarray) -> List[float]:

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))

        w = w.astype(np.float32)

        return w.tolist()


class ShapExplainer:
    def __init__(
            self, 
            model, 
            x_background : np.ndarray,
            ):

        self.explainer = shap.Explainer(model.predict, x_background)
    

    def explain_instance(self, x : np.ndarray, y : np.ndarray) -> List[float]:

        w = self.explainer.shap_values(x.reshape(1, -1)).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))

        w = w.astype(np.float32)
        
        return w.tolist()


if __name__ == "__main__":

    # np.random.seed(42)

    modelpath = "./models/rocket_SwedishLeaf_1"

    dataset = modelpath.split("/")[-1].split("_")[1]

    model = SktimeClassifier()
    model.load_pretrained_model(modelpath)

    dataset = modelpath.split("/")[-1].split("_")[1]
    ucr = UcrDataset(
        name=dataset,
        float_dtype=32,
        scale=True,
        # n_dims=32,
    )

    test_x, test_y = ucr.load_split("test")

    method = "kernel"

    if method == "kernel":

        explainer = KernelShapExplainer(
            model=model,
            x_background=np.zeros((1, test_x.shape[1])),
        )

    elif method == "regular":

        explainer = ShapExplainer(
            model=model,
            x_background=np.zeros((1, test_x.shape[1])),
        )

    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    idx = 50
    sample = test_x[idx]
    sample_y = test_y[idx]
    
    w = explainer.explain_instance(sample)

    sample = sample.reshape(-1)
    w = w.reshape(-1)