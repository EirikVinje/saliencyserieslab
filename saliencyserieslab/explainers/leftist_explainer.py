from typing import Callable
import multiprocessing
import datetime
import pickle
import json
import os

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import numpy as np
import shap

from saliencyserieslab.load_sktime_classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph

class LeftistExplainer:
    def __init__(
            self, 
            random_background : np.ndarray = None,
            perturbation_ratio : float=0.4, 
            model_fn : Callable = None, 
            num_samples : int=5000, 
            segment_size : int=1, 
            sigma : float=0.1,
                 ):

        self.perturbation_ratio = perturbation_ratio
        self.random_background = random_background
        self.segment_size = segment_size
        self.num_samples = num_samples
        self.model_fn = model_fn
        self.sigma = sigma


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
                x_perturb = x_copy.copy()

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

        x[start_idx:end_idx] = np.linspace(x[start_idx], x[end_idx], end_idx - start_idx)
        return x
    

    def perturb_constant(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        x[start_idx:end_idx] = np.repeat(x[start_idx], end_idx - start_idx)
        return x
    

    def perturb_random_background(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        random_backgroung = np.random.choice(self.random_background)
        x[start_idx:end_idx] = random_backgroung[start_idx:end_idx]
        return x


    def explain_instance(
            self,
            x : np.ndarray,
            explainer : str,
    ):
        
        """
        Explains an instance x with either "lime" or "kernelshap".

        :param x: The instance to explain.
        :param explainer: The explainer to use, either "lime" or "kernelshap".
        :return: The explanation.
        
        """

        perturbed_data, binary_rep = self._make_perturbed_data(x)

        if explainer == "lime":
            
            distances = pairwise_distances(perturbed_data, x.reshape(1, -1), metric='euclidean')

            apply_similarity = np.vectorize(lambda x: np.exp(-(x ** 2) / (2 * self.sigma ** 2)))
            distances = apply_similarity(distances).reshape(-1)

            logreg = LogisticRegression(solver='lbfgs', n_jobs=multiprocessing.cpu_count()-2, max_iter=1000)
            
            perturbed_predictions = self.model_fn(perturbed_data)
            
            logreg.fit(binary_rep, perturbed_predictions, sample_weight=distances)

            w = logreg.coef_[0].reshape(-1)

            w = np.interp(w, (w.min(), w.max()), (0, 1))
            w = np.repeat(w, x.shape[0] // w.shape[0])

            explanation = w

            return explanation
        
        elif explainer == "shap":
            
            perturbed_data = shap.sample(perturbed_data, 100)
            shap_explainer = shap.KernelExplainer(self.model_fn, perturbed_data, algorithm="linear")
            w = shap_explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
            w = np.interp(w, (w.min(), w.max()), (0, 1))
            return w


if __name__ == "__main__":
    
    model = SktimeClassifier()

    model.load_pretrained_model("./models/inception_1")

    datapath = "./data/insectsound/insectsound_test_n10.pkl"
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    print("loaded {} instances from : {}".format(data['x'].shape[0], datapath))

    explainer = LeftistExplainer(
        model_fn=model.predict, 
        perturbation_ratio=0.5,
        num_samples=20_000,
        segment_size=20,
        sigma=0.1,
        )
    
    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    sample = data['x'][0]
    
    print("explaining sample : {}".format(sample.shape))

    method = "shap"

    w = explainer.explain_instance(sample, method)

    w = w.reshape(-1)
    sample = sample.reshape(-1)

    plot_weighted_graph(sample, w, f"./plots/leftist_{method}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
