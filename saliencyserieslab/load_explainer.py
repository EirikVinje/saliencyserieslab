from typing import Callable
import json
import os

import numpy as np

from shappy import KernelShapExplainer
from shappy import TimeShapExplainer
from shappy import RegularShapExplainer

from lemon import LemonExplainer


class Explainer:
    def __init__(self):
        self.explainer = None
        self.name = None
    
    
    def load_explainer(self, explainer : str, model_fn : Callable, background_data : np.ndarray = None):

        if explainer == 'lime':
            self.load_lime(model_fn=model_fn)

        elif explainer == 'kernelshap':
            self.load_kernelshap(model_fn=model_fn, background_data=background_data)
        
        else:
            raise NotImplementedError

    
    def _load_config(self, config_path : str):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config


    def load_lime(self, model_fn : Callable, config_path : str = './explainerconfigs/lime_config.json'):
        
        config = self._load_config(config_path)
        self.explainer = LemonExplainer(model_fn=model_fn, 
                                        perturbation_ratio=config['perturbation_ratio'],
                                        adjacency_prob=config['adjacency_prob'],
                                        num_samples=config['num_samples'],
                                        segment_size=config['segment_size'],
                                        sigma=config['sigma']
                                        )
        
        self.name = self.explainer.__class__.__name__

    def load_kernelshap(self, model_fn : Callable, background_data : np.ndarray, config_path : str = './explainerconfigs/kernelshap_config.json'):
        
        config = self._load_config(config_path)
        self.explainer = KernelShapExplainer(model_fn=model_fn, 
                                      x_background=background_data,
                                      algorithm_idx=config['algorithm_idx']
                                      )
        
        self.name = self.explainer.__class__.__name__
