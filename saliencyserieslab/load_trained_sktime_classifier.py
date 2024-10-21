import sys
import os

from sklearn.metrics import accuracy_score
from sktime.utils import mlflow_sktime
import numpy as np


class SktimeClassifier:
    def __init__(self, model_path):
        
        self.model = mlflow_sktime.load_model(model_path)
        self.model_name = self.model.__class__.__name__
        self.model.verbose = False

    def predict(self, x : np.ndarray):
        
        assert len(x.shape) == 2, "Input must be a 2D numpy array"

        sys.stdout = open(os.devnull, 'w')
        predictions = self.model.predict(x)
        sys.stdout = sys.__stdout__

        return predictions
    

    def evaluate(self, x : np.ndarray, y : np.ndarray, metric : str='accuracy'):
        
        sys.stdout = open(os.devnull, 'w')
        predictions = self.predict(x)
        sys.stdout = sys.__stdout__
        
        if metric == 'accuracy':
            accuracy = accuracy_score(predictions, y)
            return accuracy

