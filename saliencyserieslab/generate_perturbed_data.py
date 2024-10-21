from typing import List
import multiprocessing
import argparse
import logging
import pickle
import json
import os

import numpy as np

from plot import plot_weighted_graph

class PerturbedDataGenerator:
    def __init__(self, explanations : np.ndarray, X : np.ndarray):
        
        """
        Perturbes the input data given a model and explainer.

        1. Perturb the input data based on the explanations            
            1. Sort explanation indices based on weight 
            2. Select top k percent indices to perturb
            
        3. Return perturbed data

        :param The trained model model: 
        :param The explanations (2D numpy array) explanations: 
        :param The original data (2D numpy array) X: 
        :param The original labels (1D numpy array) Y:
        """

        self.W = explanations
        self.X = X


    def _perturb_data(self, data_to_perturb : np.ndarray, method : str='local_mean'):

        if method == 'local_mean':

            data_to_perturb[:] = np.mean(data_to_perturb)
            return data_to_perturb 

        elif method == 'local_gaussian':
            
            mean = np.mean(data_to_perturb)
            std = np.std(data_to_perturb)
            data_to_perturb[:] = np.random.normal(mean, std, data_to_perturb.shape[0])
            return data_to_perturb        


        
    def generate(self, k : float=0.1, method : str='local_mean'):

        # Sort explanation indices based on weight
        sorted_idx = np.argsort(self.W, axis=1)[:, ::-1]

        # Select top k percent indices to perturb
        top_idx = int(k * self.W.shape[1])
        perturbed_idx = sorted_idx[:, :top_idx]

        # Perturb the input data
        perturbed_data = self.X.copy()

        for i in range(perturbed_data.shape[0]):
            perturbed_data[i, perturbed_idx[i]] = self._perturb_data(perturbed_data[i, perturbed_idx[i]], method)

        return perturbed_data
    

if __name__ == "__main__":

    with open("./data/insectsound/insectsound_test.pkl", 'rb') as f:
        evaldata = pickle.load(f)
    
    sample_x = evaldata["x"][0]
    sample_w = np.full(sample_x.shape[0], 0.3)

    sample_w[100:200] = 0.7

    generator = PerturbedDataGenerator(sample_w.reshape(1, -1), sample_x.reshape(1,-1))
    
    plot_weighted_graph(sample_x.reshape(-1), sample_w.reshape(-1), "./plots/before_perturb.png")
    perturbed_data = generator.generate(k=0.17, method="local_gaussian")
    plot_weighted_graph(perturbed_data.reshape(-1), sample_w.reshape(-1), "./plots/after_perturb.png")