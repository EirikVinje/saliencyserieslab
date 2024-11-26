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


