from typing import List
import multiprocessing
import argparse
import logging
import pickle
import json
import os

import numpy as np

class PerturbedDataGenerator:
    def __init__(self):
        
        """
        Perturbes the input data given a model and explainer.

        1. Perturb the input data based on the explanations            
            1. Sort explanation indices based on weight 
            2. Select top k percent indices to perturb
            
        3. Return perturbed data
        """
        pass

        
    def __call__(
            self,
            X : np.ndarray,
            W : np.ndarray, 
            method : str,
            k : float, 
            ):
        
        """
        :param The trained model model: 
        :param The explanations (2D numpy array) explanations: 
        :param The original data (2D numpy array) X: 
        :param The original labels (1D numpy array) Y:
        """

        if k == 0.0:
            return X

        # Sort explanation indices based on weight
        sorted_idx = np.argsort(W, axis=1)[:, ::-1]

        # Select top k percent indices to perturb
        top_idx = int(k * W.shape[1])
        
        perturbed_idx = sorted_idx[:, :top_idx]

        # Perturb the input data
        perturbed_data = X.copy()

        for i in range(perturbed_data.shape[0]):
            perturbed_data[i, perturbed_idx[i]] = self._perturb_data(perturbed_data[i, perturbed_idx[i]], perturbed_data[i], method)

        return perturbed_data
    

    def _perturb_data(self, data_to_perturb : np.ndarray, orig_data : np.ndarray, method : str='local_mean'):
    

        if method == 'local_mean':
            
            data_to_perturb[:] = np.mean(data_to_perturb)
            return data_to_perturb 

        elif method == 'global_mean':
            
            data_to_perturb[:] = np.mean(orig_data)
            return data_to_perturb
        
        elif method == 'local_gaussian':

            data_to_perturb[:] = np.random.normal(np.mean(data_to_perturb), scale=np.std(data_to_perturb), size=data_to_perturb.shape[0])
            return data_to_perturb

        elif method == 'global_gaussian':
            
            data_to_perturb[:] = np.random.normal(np.mean(orig_data), scale=np.std(orig_data), size=data_to_perturb.shape[0])
            return data_to_perturb