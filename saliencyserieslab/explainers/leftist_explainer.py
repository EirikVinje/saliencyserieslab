from typing import Callable
import multiprocessing
import datetime
import pickle
import json
import os

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression, LinearRegression
from aeon.datasets import load_classification
from scipy.spatial.distance import euclidean
from scipy import signal
from tqdm import tqdm
import numpy as np
import shap

from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph, plot_graph, plot_simple_weighted

class LeftistExplainer:
    def __init__(
            self, 
            model, 
            random_background : np.ndarray = None,
            perturbation_ratio : float=0.4, 
            n_reduce_dims : int = None,
            num_samples : int=5000, 
            segment_size : int=1,
            method : str = "lime",
            ):

        self.perturbation_ratio = perturbation_ratio
        self.random_background = random_background
        self.n_reduce_dims = n_reduce_dims
        self.segment_size = segment_size
        self.num_samples = num_samples
        self.method = method
        self.model = model
        

    def _make_perturbed_data(self, x : np.array, progress_bar : bool = True):

        assert len(x.shape) == 1, "reshape x to : (n,)"

        x_copy = x.copy()
        
        mod = x_copy.shape[0] % self.segment_size
        if mod != 0:
            x_copy = x_copy[:-mod]
        
        num_segments = x_copy.shape[0] // self.segment_size
        
        perturbed_matrix = np.random.choice(
            [0, 1], 
            size=(self.num_samples, num_segments), 
            p=[self.perturbation_ratio, 1-self.perturbation_ratio]
            )
        
        perturbed_data = []
        
        with tqdm(total=int(self.num_samples*num_segments), disable=not progress_bar, desc="Perturbing data") as bar:

            for i in range(self.num_samples):
                x_perturb = x.copy()

                for j in range(num_segments):
                        
                    if perturbed_matrix[i, j] == 0:
                        
                        start = j * self.segment_size
                        end = start + self.segment_size

                        method = np.random.choice(["linear_interpolation", "constant", "random_background"])

                        if method == 'linear_interpolation':
                            x_perturb = self._perturb_linear_interpolation(x_perturb, start, end)
                        elif method == 'constant':
                            x_perturb = self._perturb_constant(x_perturb, start, end)
                        elif method == 'random_background':
                            x_perturb = self._perturb_random_background(x_perturb, start, end)

                    bar.update(1)

                perturbed_data.append(x_perturb)

        perturbed_data = np.vstack(perturbed_data)

        perturbed_binary = perturbed_matrix 

        return perturbed_data, perturbed_binary


    def _perturb_linear_interpolation(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        if end_idx >= x.shape[0]:
            end_idx = x.shape[0]-1

        perturbation = np.linspace(x[start_idx], x[end_idx], end_idx - start_idx)
        x[start_idx:end_idx] = perturbation
        return x
    

    def _perturb_constant(self, x : np.ndarray, start_idx : int, end_idx : int):

        perturbation = np.repeat(x[start_idx], end_idx - start_idx)
        x[start_idx:end_idx] = perturbation
        return x
    

    def _perturb_random_background(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        rand_idx = np.random.choice(range(self.random_background.shape[0]))
        perturbation = self.random_background[rand_idx, start_idx:end_idx]
        
        x[start_idx:end_idx] = perturbation
        return x


    def explain_instance(
            self,
            x : np.ndarray,
    ):
        
        """
        Explains an instance x with either "lime" or "kernelshap".

        :param x: The instance to explain.
        :param explainer: The explainer to use, either "lime" or "kernelshap".
        :return: The explanation.
        
        """

        if self.method == "lime":

            perturbed_data, binary_rep = self._make_perturbed_data(x)
            
            distances = pairwise_distances(perturbed_data, x.reshape(1, -1), metric='euclidean')
            distances = np.interp(distances, (distances.min(), distances.max()), (0, 1)).reshape(-1)
  
            perturbed_predictions = self.model.predict(perturbed_data)
            
            logreg = LogisticRegression(
                solver='lbfgs', 
                n_jobs=multiprocessing.cpu_count()-2, 
                max_iter=1000
                )
            logreg.fit(binary_rep, perturbed_predictions, sample_weight=distances)

            explanation = logreg.coef_[0].reshape(-1)

            explanation = np.interp(explanation, (explanation.min(), explanation.max()), (0, 1))
            explanation = np.repeat(explanation, x.shape[0] // explanation.shape[0])

            return explanation
        
        elif self.method == "shap": 

            x = (x - x.mean()) / x.std()
            
            perturbed_data, _ = self._make_perturbed_data(x)
            perturbed_data = shap.sample(perturbed_data, 10)

            print("perturbed data shape : {}".format(perturbed_data.shape))
            print("x shape : {}".format(x.shape))
            
            shap_explainer = shap.KernelExplainer(self.model.predict, perturbed_data)
            w = shap_explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)

            print(w)
            
            w = np.interp(w, (w.min(), w.max()), (0, 1))

            print("w shape: ", w.shape)
            
            return w


    


if __name__ == "__main__":
    
    np.random.seed(42)

    modelpath = "./models/rocket_ECG200_1"
    dataset = modelpath.split("/")[-1].split("_")[1]

    model = SktimeClassifier()
    model.load_pretrained_model(modelpath)

    test = load_classification(dataset, split="test")

    unique_classes = np.unique(test[1]).tolist()
    unique_classes = [int(c) for c in unique_classes]
    unique_classes.sort()
 
    test = {
        "x" : test[0].squeeze(),
        "y" : np.array([unique_classes.index(int(c)) for c in test[1]]),
    }

    method = "shap"
    explainer = LeftistExplainer(
        random_background=test["x"],
        perturbation_ratio=0.5,
        num_samples=1_000,
        n_reduce_dims=48,
        segment_size=3,
        method=method,
        model=model, 
        )
    
    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    sample = test["x"][np.random.randint(0, test["x"].shape[0])]


    w = explainer.explain_instance(sample)

    sample = sample.reshape(-1)
    w = w.reshape(-1)

    plot_simple_weighted(
        sample, 
        w, 
        f"./plots/simple_leftist_{method}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png",
        model.model.__class__.__name__,
        explainer.__class__.__name__,
        dataset
        )
