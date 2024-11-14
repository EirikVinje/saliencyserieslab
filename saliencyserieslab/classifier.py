import multiprocessing
import json
import sys
import os

from sklearn.metrics import accuracy_score
from sktime.utils import mlflow_sktime
import numpy as np

# get softmax
import keras

# from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.distance_based import ProximityForest
from sktime.classification.deep_learning import ResNetClassifier

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.shapelet_based import MrSEQL
# from mrsqm import MrSQMClassifier



N_JOBS = multiprocessing.cpu_count() - 2

class SktimeClassifier:
    def __init__(self):
        
        self.model = None
        self.name = None


    def load_pretrained_model(self, model_path : str):
        
        self.model = mlflow_sktime.load_model(model_path)
        self.name = self.model.__class__.__name__
        self.model.verbose = False

        if model_path.split("/")[-1].split("_")[0] == "weasel":
            self.model.support_probabilities = True


    def fit(self, x : np.ndarray, y : np.ndarray):
        self.model.fit(x, y)


    def load_model(self, model : str):

        if model == 'inception':
            self._load_inception()
        elif model == 'rocket':
            self._load_rocket()
        elif model == 'resnet':
            self._load_resnet()
        elif model == 'mrseql':
            self._load_mrseql()
        elif model == 'proximityforest':
            self._load_proximityforest()
        elif model == 'weasel':
            self._load_weasel()
        else:
            raise ValueError('Invalid model name')
    

    def predict(self, x : np.ndarray, verbose : bool = False):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if verbose:
            self.model.verbose = True
            predictions = self.model.predict(x)
            return predictions
        
        else:
            sys.stdout = open(os.devnull, 'w')
            predictions = self.model.predict(x)
            sys.stdout = sys.__stdout__
            return predictions


    def predict_proba(self, x : np.ndarray, verbose : bool = False):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if verbose:
            self.model.verbose = True
            predictions = self.model.predict_proba(x)
            return predictions
        
        else:
            sys.stdout = open(os.devnull, 'w')
            predictions = self.model.predict_proba(x)
            sys.stdout = sys.__stdout__
            return predictions
    

    def evaluate(self, x : np.ndarray, y : np.ndarray, metric : str='accuracy'):
        
        sys.stdout = open(os.devnull, 'w')
        predictions = self.predict(x)
        sys.stdout = sys.__stdout__
        
        if metric == 'accuracy':
            accuracy = accuracy_score(predictions, y)
            return accuracy
        

    def __call__(self, x : np.ndarray) :
        return self.predict(x)


    def _load_resnet(self):

        self.model = ResNetClassifier(
            random_state=42,
            verbose=True, 
            n_epochs=20, 
            )
        
        self.name = self.model.__class__.__name__


    def _load_inception(self):

        self.model = InceptionTimeClassifier(
            random_state=42,
            verbose=True, 
            n_epochs=20,
            )
        
        self.name = self.model.__class__.__name__


    def _load_rocket(self):

        self.model = RocketClassifier(
            random_state=42,
            n_jobs=N_JOBS,
            )
        
        self.name = self.model.__class__.__name__


    def _load_mrseql(self):

        self.model = MrSEQL()
        
        self.name = self.model.__class__.__name__


    def _load_weasel(self):

        self.model = WEASEL(
            n_jobs=N_JOBS,
            random_state=42,
            )
        
        self.name = self.model.__class__.__name__

    
    def _load_proximityforest(self):

        self.model = ProximityForest(
            n_jobs=N_JOBS,
            random_state=42,
            n_estimators=1,
            verbosity=True,
            max_depth=1,
            )
        
        self.name = self.model.__class__.__name__



class Weasel:

    def __init__(self):
        
        self.model = None
        self.name = None


    def load_pretrained_model(self, model_path : str):
        
        self.model = mlflow_sktime.load_model(model_path)
        self.name = self.model.__class__.__name__
        self.model.support_probabilities=True


    def predict(self, x : np.ndarray):
        return self.model.predict(x)


    def predict_proba(self, x : np.ndarray):

        scores = self.model.decision_function(x)

        scores = keras.activations.softmax(scores)

        print(scores)



    def explain_instance(self, x : np.ndarray):
        
        explanation = self.model.get_saliency_map(x)
        print(explanation)


