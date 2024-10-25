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


class SktimeClassifier:
    def __init__(self):
        self.model = None
        self.name = None
        self.config = None

    def load_pretrained_model(self, model_path : str):
        
        self.model = mlflow_sktime.load_model(model_path)
        self.name = self.model.__class__.__name__
        self.model.verbose = False


    def load_model(self, model : str):

        if model == 'inception':
            self._load_inception()
        elif model == 'rocket':
            self._load_rocket()
        elif model == 'resnet':
            self._load_resnet()
        else:
            raise ValueError('Invalid model name')
    

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


    def _load_config(self, config_path : str):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config    
    
        return config


    def _load_resnet(self, config_path : str = './modelconfigs/resnet_config.json'):

        config = self._load_config(config_path)
        self.model = ResNetClassifier(n_epochs=config['epochs'], 
                                      verbose=config['verbose'], 
                                      batch_size=config['batch_size'],
                                      random_state=config['random_state'])
        
        self.name = self.model.__class__.__name__


    def _load_inception(self, config_path : str = './modelconfigs/inception_config.json'):

        config = self._load_config(config_path)
        self.model = InceptionTimeClassifier(n_epochs=config['epochs'], 
                                             verbose=config['verbose'], 
                                             batch_size=config['batch_size'],
                                             random_state=config['random_state'])
        
        self.name = self.model.__class__.__name__



    def _load_rocket(self, config_path : str = './modelconfigs/rocket_config.json'):

        config = self._load_config(config_path)
        self.model = RocketClassifier(num_kernels=config['num_kernels'],
                                      rocket_transform=config['rocket_transform'],
                                      use_multivariate="no", 
                                      n_jobs=multiprocessing.cpu_count()-2,
                                      random_state=config['random_state'])
        
        self.name = self.model.__class__.__name__
