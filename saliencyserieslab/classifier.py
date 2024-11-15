import multiprocessing
import pickle
import sys
import os

from sklearn.metrics import accuracy_score
from sktime.utils import mlflow_sktime
import numpy as np

from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning import ResNetClassifier
from sktime.classification.kernel_based import RocketClassifier
from mrseql import MrSEQLClassifier


N_JOBS = multiprocessing.cpu_count() - 2


class SktimeClassifier:
    def __init__(self):
        
        self.model = None
        self.name = None


    def load_pretrained_model(self, model_path : str):
        
        if model_path.split("/")[-1].split("_")[0] == "mrseql":
            
            modelpath_pkl = os.path.join(model_path, model_path.split("/")[-1] + ".pkl") 

            with open(modelpath_pkl, 'rb') as f:
                self.model = pickle.load(f)
            
        else:
            self.model = mlflow_sktime.load_model(model_path)
        
        self.name = self.model.__class__.__name__


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
    

    def predict(self, x : np.ndarray):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        sys.stdout = open(os.devnull, 'w')
        predictions = self.model.predict(x)
        sys.stdout = sys.__stdout__
        return predictions


    def predict_proba(self, x : np.ndarray):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
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
        

    def _load_inception(self):

        self.model = InceptionTimeClassifier(
            random_state=42,
            batch_size=8,
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

        self.model = MrSEQLClassifier()
        self.name = self.model.__class__.__name__


    def _load_resnet(self):

        self.model = ResNetClassifier(
            n_epochs=25,
            random_state=42,
            verbose=False, 
            batch_size=8,
            )
        
        self.name = self.model.__class__.__name__




