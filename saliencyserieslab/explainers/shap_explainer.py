from typing import Callable
import datetime
import pickle
import json
import os

import numpy as np
import shap

from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph


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
    
    pass