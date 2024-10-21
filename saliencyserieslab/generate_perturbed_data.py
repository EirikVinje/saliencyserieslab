from typing import List
import multiprocessing
import argparse
import logging
import pickle
import json
import os

import numpy as np

class PerturbedDataGenerator:
    def __init__(self, explanations : list[np.ndarray, int], X : np.ndarray, Y : np.ndarray):
        
        """
        Perturbes the input data given a model and explainer.

        1. Perturb the input data based on the explanations            
            1. Sort explanation indices based on weight 
            2. Select top k percent indices to perturb
            
        3. Return perturbed data 
        
        :param model: The trained model
        :param explanations: The explanations (2D numpy array)
        :param X: The original data (2D numpy array)
        :param Y: The original labels (1D numpy array)
        """

        self.explanations = np.array([exp[0] for exp in explanations])
        indices = [exp[1] for exp in explanations]
        
        self.X = X[indices]
        self.Y = Y[indices]
    

    def _perturb_data(self, data_to_perturb : np.ndarray, method : str='local_mean'):

        if method == 'local_mean':

            data_to_perturb[:] = np.mean(data_to_perturb)
            return data_to_perturb 

        if method == 'local_gaussian':

            mean = np.mean(data_to_perturb)
            std = np.std(data_to_perturb)
            data_to_perturb[:] = np.random.normal(mean, std, data_to_perturb.shape)

            return data_to_perturb        

        
    def generate(self, k : float=0.1, method : str='local_mean'):

        # Sort explanation indices based on weight
        sorted_idx = np.argsort(self.explanations, axis=1)[:, ::-1]

        # Select top k percent indices to perturb
        top_idx = int(k * self.explanations.shape[1])
        perturbed_idx = sorted_idx[:, :top_idx]

        # Perturb the input data
        perturbed_data = self.X.copy()

        for i in range(perturbed_data.shape[0]):
            perturbed_data[i, perturbed_idx[i]] = self._perturb_data(perturbed_data[i, perturbed_idx[i]], method)

        return perturbed_data, self.Y