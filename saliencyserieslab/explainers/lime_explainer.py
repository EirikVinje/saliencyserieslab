from typing import Callable, List
import multiprocessing
import datetime
import pickle
import json
import os

from tqdm import tqdm
import numpy as np
import shap

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression


from saliencyserieslab.classifier import SktimeClassifier


class LimeExplainer:
    def __init__(
            self, 
            model, 
            perturbation_ratio : float=0.6, 
            num_samples : int=2500,
            progress_bar : bool=False
            ):

        self.perturbation_ratio = perturbation_ratio
        self.num_samples = num_samples
        self.model = model
        self.progress_bar = progress_bar
        

    def _make_perturbed_data(self, x : np.array, progress_bar : bool = True):

        assert len(x.shape) == 1, "reshape x to : (n,)"

        x_copy = x.copy()
        
        perturbed_matrix = np.random.choice(
            [0, 1], 
            size=(self.num_samples, x_copy.shape[0]), 
            p=[self.perturbation_ratio, 1-self.perturbation_ratio]
            )
        
        perturbed_data = []
        
        with tqdm(total=int(self.num_samples*x_copy.shape[0]), disable=not self.progress_bar, desc="Perturbing data") as bar:

            for i in range(self.num_samples):
                x_perturb = x_copy.copy()

                for j in range(x_copy.shape[0]):
                        
                    if perturbed_matrix[i, j] == 0:

                        method = np.random.choice(['zero', 'noise', 'total_mean'])
                
                        if method == "zero":
                            x_perturb[j] = 0
                        elif method == "noise":
                            x_perturb[j] += np.random.normal(0, scale=np.std(x), size=1)[0]
                        elif method == "total_mean":
                            x_perturb[j] = np.mean(x)

                    bar.update(1)

                perturbed_data.append(x_perturb)

        perturbed_data = np.vstack(perturbed_data)

        perturbed_binary = perturbed_matrix 

        return perturbed_data, perturbed_binary

    
    def explain_instance(self, x : np.ndarray, y : np.ndarray,) -> List[float]:

        perturbed_data, binary_rep = self._make_perturbed_data(x)
    
        distances = pairwise_distances(perturbed_data, x.reshape(1, -1), metric='euclidean')

        distances = np.interp(distances, (distances.min(), distances.max()), (0, 1)).reshape(-1)

        logreg = LogisticRegression(
            solver='lbfgs', 
            n_jobs=multiprocessing.cpu_count()-2, 
            max_iter=1000
            )
        
        perturbed_predictions = self.model.predict(perturbed_data)
        
        if np.unique(perturbed_predictions).shape[0] == 1:
            print("bad finish")
            return np.zeros(x.shape[0])

        logreg.fit(binary_rep, perturbed_predictions, sample_weight=distances)

        explanation = logreg.coef_[0].reshape(-1)

        w = np.interp(explanation, (explanation.min(), explanation.max()), (0, 1))

        w = w.astype(np.float32)

        return w.tolist()


