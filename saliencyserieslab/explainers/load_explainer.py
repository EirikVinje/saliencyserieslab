from typing import Callable
import multiprocessing
import json
import os

from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
import numpy as np
import shap


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
        self.explainer = LemonExplainer(
            model_fn=model_fn,
            perturbation_ratio=config['perturbation_ratio'],
            adjacency_prob=config['adjacency_prob'],
            num_samples=config['num_samples'],
            segment_size=config['segment_size'],
            sigma=config['sigma'],
            )
        
        self.name = self.explainer.__class__.__name__


    def load_kernelshap(self, model_fn : Callable, background_data : np.ndarray, config_path : str = './explainerconfigs/kernelshap_config.json'):
        
        config = self._load_config(config_path)
        self.explainer = KernelShapExplainer(
            model_fn=model_fn,
            x_background=background_data,
            algorithm_idx=config['algorithm_idx'],
            n_background=config['n_background'],
            )
        
        self.name = self.explainer.__class__.__name__


class RegularShapExplainer:
    def __init__(self, model_fn : Callable, x : np.ndarray):
        self.explainer = shap.Explainer(model_fn, x)


    def explain(self, x : np.ndarray, max_evals : int):

        shap_values = self.explainer(x, max_evals=max_evals)
        w = shap_values.values[0]
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w


class TimeShapExplainer:
    def __init__(self):
        pass


class KernelShapExplainer:
    def __init__(self, model_fn : Callable, x_background : np.ndarray, algorithm_idx : int=0, n_background : int=50):

        x_background = shap.sample(x_background, n_background)
        algorithms = ["auto", "permutation", "partition", "tree", "linear"]
        self.explainer = shap.KernelExplainer(model_fn, x_background, algorithm=algorithms[algorithm_idx])


    def explain(self, x : np.ndarray):

        w = self.explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        return w
    

class LemonExplainer:
    def __init__(self, 
                 model_fn : Callable, 
                 perturbation_ratio : float=0.4, 
                 adjacency_prob : float=0.9, 
                 num_samples : int=5000, 
                 segment_size : int=1, 
                 sigma : float=0.1
                 ):
    
        self.perturbation_ratio = perturbation_ratio
        self.adjacency_prob = adjacency_prob
        self.segment_size = segment_size
        self.num_samples = num_samples
        self.model_fn = model_fn
        self.sigma = sigma
        

    def _distance(self, original, perturbed, method='euclidean'):
        
        if method == 'dtw':
            pass
        elif method == 'euclidean':
            distance = euclidean(original, perturbed)
        elif method == 'correlation':
            correlation = np.corrcoef(original, perturbed)[0, 1]
            distance = 1 - abs(correlation)
        else:
            raise ValueError("Unsupported distance method. Choose 'dtw', 'euclidean', or 'correlation'.")
        
        similarity = np.exp(-(distance ** 2) / (2 * self.sigma ** 2))
        
        return similarity


    def _perturb_data(self, x : np.ndarray):
    
        assert len(x.shape) == 1, "Input must be a 1D numpy array"
        
        padded_length = ((x.shape[0] - 1) // self.segment_size + 1) * self.segment_size
        padded_timeseries = np.pad(x, (0, padded_length - x.shape[0]), mode='constant', constant_values=0)
        
        padded_timeseries = padded_timeseries.astype(float)
        
        num_segments = padded_length // self.segment_size
        num_perturb = int(num_segments * self.perturbation_ratio)
        
        perturbed_samples = []
        binary_representations = []
        
        for _ in range(self.num_samples):
            perturb_mask = np.zeros(num_segments, dtype=bool)
            
            current_segment = np.random.randint(0, num_segments)
            perturb_mask[current_segment] = True
            
            while np.sum(perturb_mask) < num_perturb:
                if np.random.random() < self.adjacency_prob:
                    direction = np.random.choice([-1, 1])  
                    next_segment = (current_segment + direction) % num_segments
                else:
                    unperturbed = np.where(~perturb_mask)[0]
                    next_segment = np.random.choice(unperturbed)
                
                perturb_mask[next_segment] = True
                current_segment = next_segment
            
            perturbed = np.copy(padded_timeseries)
            binary_rep = np.ones(num_segments)
            
            for idx in np.where(perturb_mask)[0]:
                start = idx * self.segment_size
                end = start + self.segment_size
                
                method = np.random.choice(['zero', 'noise', 'shuffle'])
                
                if method == 'zero':
                    perturbed[start:end] = 0
                elif method == 'noise':
                    perturbed[start:end] += np.random.normal(0, np.std(x), self.segment_size)
                elif method == 'shuffle':
                    np.random.shuffle(perturbed[start:end])
                
                binary_rep[idx] = 0
            
            perturbed_samples.append(perturbed[:x.shape[0]])
            binary_representations.append(binary_rep)
        
        return np.array(perturbed_samples), np.array(binary_representations)


    def explain(self, x : np.ndarray):
        
        x_perturb, binary_rep = self._perturb_data(x)

        weights = [self._distance(x, x_perturb[i]) for i in range(len(x_perturb))]
        
        perturb_predictions = self.model_fn(x_perturb)
        
        logreg = LogisticRegression(solver='lbfgs', n_jobs=multiprocessing.cpu_count()-2, max_iter=1000)
        logreg.fit(binary_rep, perturb_predictions, sample_weight=weights)

        w = logreg.coef_[0].reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        w = np.repeat(w, x.shape[0] // w.shape[0])
                
        return w