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
    def __init__(self, 
                 model_fn : Callable = None, 
                 perturbation_ratio : float=0.4, 
                 adjacency_prob : float=0.9, 
                 num_samples : int=5000, 
                 segment_size : int=1, 
                 sigma : float=0.1):
    
        self.perturbation_ratio = perturbation_ratio
        self.adjacency_prob = adjacency_prob
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

                        method = np.random.choice(['total_mean', 'mean', 'noise'])

                        if method == 'total_mean':
                            x_perturb = self._perturb_total_mean(x_perturb, start, end)
                        elif method == 'mean':
                            x_perturb = self._perturb_mean(x_perturb, start, end)
                        elif method == 'noise':
                            x_perturb = self._perturb_noise(x_perturb, start, end)

                    bar.update(1)

                perturbed_data.append(x_perturb)

        perturbed_data = np.vstack(perturbed_data)

        perturbed_binary = perturbed_matrix 

        return perturbed_data, perturbed_binary
    

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


    def _perturb_total_mean(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        x[start_idx:end_idx] = np.repeat(np.mean(x), end_idx - start_idx)
        return x
                

    def _perturb_mean(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        x[start_idx:end_idx] = np.repeat(np.mean(x[start_idx:end_idx]), end_idx - start_idx)
        return x
        
        
    def _perturb_noise(self, x : np.ndarray, start_idx : int, end_idx : int):
        
        x[start_idx:end_idx] = np.random.uniform(x.min(), x.max(), end_idx - start_idx)
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

        # perturbed_data, binary_rep = self._perturb_data(x)
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
            
            perturbed_data = shap.kmeans(perturbed_data, 100)

            shap_explainer = shap.KernelExplainer(self.model_fn, perturbed_data, algorithm="linear")
            w = shap_explainer.shap_values(x, gc_collect=True, silent=True).reshape(-1)
            w = np.interp(w, (w.min(), w.max()), (0, 1))
            return w


if __name__ == "__main__":
    
    model = SktimeClassifier()

    model.load_pretrained_model("./models/inception_1")

    explainer = LeftistExplainer(
        model_fn=model.predict, 
        perturbation_ratio=0.5,
        adjacency_prob=0.7,
        num_samples=20_000,
        segment_size=20,
        sigma=0.1,
        )
    
    print("loaded model and explainer : ({}, {})".format(model.__class__.__name__, explainer.__class__.__name__))

    datapath = "./data/insectsound/insectsound_test_n10.pkl"

    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    
    print("loaded {} instances from : {}".format(data['x'].shape[0], datapath))

    sample = data['x'][0]
    
    print("explaining sample : {}".format(sample.shape))

    method = "shap"

    w = explainer.explain_instance(sample, method)

    w = w.reshape(-1)
    sample = sample.reshape(-1)

    plot_weighted_graph(sample, w, f"./plots/{method}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")