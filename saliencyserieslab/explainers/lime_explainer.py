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


class LimeExplainer:
    def __init__(
            self, 
            random_background : np.ndarray = None,
            perturbation_ratio : float=0.4, 
            model_fn : Callable = None, 
            adjacency_prob : float=0.9, 
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
        
        perturbed_matrix = np.random.choice(
            [0, 1], 
            size=(self.num_samples, x_copy.shape[1]), 
            p=[self.perturbation_ratio, 1-self.perturbation_ratio]
            )
        
        perturbed_data = []
        
        with tqdm(total=int(self.num_samples*x_copy.shape[1]), disable=not progress_bar, desc="Perturbing data") as bar:

            for i in range(self.num_samples):
                x_perturb = x_copy.copy()

                for j in range(x_copy.shape[1]):
                        
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

    
    def explain_instance(
            self,
            x : np.ndarray,
    ):
        
        """
        Explains an instance x .

        :param x: The instance to explain.
        :return: The explanation.
        
        """

        perturbed_data, binary_rep = self._make_perturbed_data(x)
    
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


if __name__ == "__main__":
    
    model = SktimeClassifier()

    model.load_pretrained_model("./models/inception_1")

    datapath = "./data/insectsound/insectsound_test_n10.pkl"
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    print("loaded {} instances from : {}".format(data['x'].shape[0], datapath))

    explainer = LimeExplainer(
        model_fn=model.predict, 
        perturbation_ratio=0.5,
        num_samples=20_000,
        segment_size=20,
        sigma=0.1,
        )
    
    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))
    
    sample = data['x'][0]
    
    print("explaining sample : {}".format(sample.shape))

    w = explainer.explain_instance(sample)

    w = w.reshape(-1)
    sample = sample.reshape(-1)

    plot_weighted_graph(sample, w, f"./plots/lime_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
