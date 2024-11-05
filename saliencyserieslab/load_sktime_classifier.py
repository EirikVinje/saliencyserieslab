import multiprocessing
import json
import sys
import os

from sklearn.metrics import accuracy_score
from sktime.utils import mlflow_sktime
import numpy as np
# from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning import ResNetClassifier
from sktime.classification.kernel_based import RocketClassifier

# get mrseql
from sktime.classification.shapelet_based import MrSEQL


class SktimeClassifier:
    def __init__(self, config : dict = None):
        
        self.config = config
        
        self.model = None
        self.name = None

    def load_pretrained_model(self, model_path : str):
        
        self.model = mlflow_sktime.load_model(model_path)
        self.name = self.model.__class__.__name__
        self.model.verbose = False


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
        else:
            raise ValueError('Invalid model name')
    

    def predict(self, x : np.ndarray, verbose : bool = True):
        
        assert len(x.shape) == 2, "Input must be a 2D numpy array"

        if verbose:
            self.model.verbose = True
            predictions = self.model.predict(x)
            return predictions
        
        else:
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
        

    def _load_resnet(self):

        config = self.config["resnet"]
        self.model = ResNetClassifier(
            random_state=config['random_state'],
            batch_size=config['batch_size'],
            n_epochs=config['epochs'], 
            verbose=config['verbose'], 
            )
        
        self.name = self.model.__class__.__name__


    def _load_inception(self):

        config = self.config["inception"]
        self.model = InceptionTimeClassifier(
            bottleneck_size=config['bottleneck_size'],
            random_state=config['random_state'],
            kernel_size=config['kernel_size'],
            batch_size=config['batch_size'],
            n_filters=config['n_filters'],
            n_epochs=config['epochs'],
            verbose=config['verbose'],
            depth=config['depth'],
            )
        
        self.name = self.model.__class__.__name__


    def _load_rocket(self):

        config = self.config["rocket"]
        self.model = RocketClassifier(
            max_dilations_per_kernel=config['max_dilations_per_kernel'],
            n_features_per_kernel=config['n_features_per_kernel'],
            rocket_transform=config['rocket_transform'],
            n_jobs=multiprocessing.cpu_count()-2,
            random_state=config['random_state'],
            num_kernels=config['num_kernels'],
            use_multivariate="no", 
            )
        
        self.name = self.model.__class__.__name__

    def _load_mrseql(self):

        self.model = MrSEQL(seql_mode="clf")
        self.name = self.model.__class__.__name__
