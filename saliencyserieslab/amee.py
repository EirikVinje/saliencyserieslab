from typing import Tuple, Callable
import argparse
import logging
import pickle
from copy import deepcopy
import json
import os

import subprocess

from tqdm import tqdm
import numpy as np

from saliencyserieslab.amee_perturbed_data import PerturbedDataGenerator
from saliencyserieslab.amee_explanations import generate_explanations
from saliencyserieslab.classifier import SktimeClassifier

# logger = logging.getLogger('saliencyserieslab')
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


def recommender(models : Tuple, explainers : Tuple, test_x : np.ndarray):

    """
    Generates explanation AUC scores for each model and explainer pair and ranks them accordingly.

    :param Tuple, The trained models, e.g (SktimeClassifier(), SktimeClassifier(), ...) models:
    :param Tuple, The explainer, e.g (Explainer(), Explainer(), ...) explainers:
    :param np.ndarray, The test set (x) test_x:

    """
    
    all_eauc = []

    for method in ["local_mean", "global_mean"]:

        for model in models: 

            model_eauc = []

            for explainer in explainers:
                
                # logger.info(f"calculating model {model.name} with explainer {explainer.name} with method {method}")

                W = generate_explanations(model, explainer, test_x)
                
                generator = PerturbedDataGenerator(W, test_x)

                accuracy = []
                k = np.arange(0.0, 1.01, 0.1)
                
                for i, ki in enumerate(k):
                    perturbed_x, y = generator.generate(ki, method=method)
                    accuracy.append(model.evaluate(perturbed_x, y))

                eauc = np.trapz(x=k, y=accuracy) 
                model_eauc.append(eauc)

            model_eauc_normalized = np.interp(model_eauc, (min(model_eauc), max(model_eauc)), (0, 1)).round(decimals=3)
        
        all_eauc.append(model_eauc_normalized)
     
    all_eauc = np.vstack(all_eauc)
    avg_eauc = np.mean(all_eauc, axis=0)
    eauc_rank = np.interp(avg_eauc, (avg_eauc.min(), avg_eauc.max()), (0, 1))
    exp_power = 1 - eauc_rank

    exp_power = {str(method.__class__.__name__) : exp_power[i] for i, method in enumerate(explainers)}

    return exp_power


def recommender_2(exp_dir : str, test_x : np.ndarray):
    
    exp_dir = os.path.join("./explanations", exp_dir)

    weights_id = os.listdir(exp_dir)
    weights = deepcopy(weights_id)

    weights_id = [w for w in weights_id if os.path.isfile(os.path.join(exp_dir, w)) and w.endswith('.pkl')]
    weights_id = [w.split('.')[0] for w in weights_id]
    weights_id = [w.split('_') for w in weights_id]

    record_euc = {"local_mean" : {}, "global_mean" : {}}

    with tqdm(total=len(models) * len(explainers)) as bar:
    
        for w_path, w_id in zip(weights, weights_id):
            
            modelname = w_id[0]
            explainername = w_id[1]
            run_id = w_id[2]
            
            bar.set_description("calculating eauc for pair : ({}, {})".format(modelname, explainername))

            modelpath = "./models/{}_{}".format(modelname, run_id)

            model = SktimeClassifier()
            model.load_pretrained_model(modelpath)

            with open(os.path.join(exp_dir, w_path), 'rb') as f:
                W = pickle.load(f)

            generator = PerturbedDataGenerator(W, test_x)

            for method in record_euc.keys():
            
                accuracy = []
                k = np.arange(0.0, 1.01, 0.1)
                for i, ki in enumerate(k):
                    perturbed_x, y = generator.generate(ki, method=method)
                    accuracy.append(model.evaluate(perturbed_x, y))

                eauc = np.trapz(x=k, y=accuracy) 

                if modelname not in record_euc[method]:
                    record_euc[method][modelname] = {}

                record_euc[method][modelname][explainername] = eauc
                
    return record_euc

if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--expdir', type=str, default='exp_1', help='Path to eval data')
    args = parser.parse_args()

    testpath = args.testpath
    
    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)
    
    models = ["inception", "rocket", "resnet"]
    explainers = ["lime", "kernelshap"]

    models = [SktimeClassifier() for name in models]
    
    test_x = evaldata['x'][0]
    exp_power = recommender(models, explainers, test_x)
    print(exp_power)
