from typing import Dict, Callable, List
import warnings
import argparse
import datetime
import logging
import pickle
import json
import os
import gc

from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from tqdm import tqdm

from saliencyserieslab.amee_classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph
from saliencyserieslab.explainers.load_explainer import Explainer

# logger = logging.getLogger('src')
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

def explain(
        explainer : Callable, 
        test_x : np.ndarray,
        explainer_name : str,
        model_name : str,
        savedir : str
        ):

    """
    Given a a pair (model, explainer), generates the \n 
    explanation "w" for each xi in test_x.

    :param Callable, The trained model model: 
    :param Callable, The explainer explainer: 
    :param np.ndarray, The test set (x) test_x: 
    :param str, The name of the explainer explainer_name: 
    :param str, The name of the model model_name: 
    :param str, The id run_id:

    :return np.ndarray: 
    """

    w_path = os.path.join(savedir, f"{model_name}_{explainer_name}.pkl")
    
    if os.path.exists(w_path):
        raise RuntimeError("explanation weights already exists at {}".format(w_path))
    
    w = []
    
    for i in tqdm(range(test_x.shape[0]), desc='generating explanations', leave=False):
        wi = explainer(test_x[i])
        w.append(wi)

    w = np.vstack(w)

    with open(w_path, 'wb') as f:
        pickle.dump(w, f)


def generate_explanations(models : List, explainers : List, test_x : np.ndarray, savedir : str):

    with tqdm(total=len(models) * len(explainers)) as bar:

        for modelname in models:
            
            model_path = f"./models/{modelname}"
            model = SktimeClassifier()
            model.load_pretrained_model(model_path)
            
            for explainername in explainers:
                bar.set_description("generating weights for pair : ({}, {})".format(modelname, explainer))
                
                explainer = Explainer()
                explainer.load_explainer(explainername, model.predict, test_x)

                explain(explainer.explain, test_x, explainername, modelname, savedir)

                explainer = None
                gc.collect()    

                bar.update(1)
            
            model = None
            gc.collect()


if __name__ == "__main__":

    print()
    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--savedir', type=str, default='w1', help='Path to eval data')
    args = parser.parse_args()
    
    savedir = "./explanations/{}".format(args.savedir)

    if os.path.isdir(savedir):
        raise RuntimeError("savedir {} already exists. Change the savedir or delete the directory".format(savedir))
    else:
        os.makedirs(savedir)

    print("weights savingdir : {}".format(savedir))
    
    with open(args.testpath, 'rb') as f:
        evaldata = pickle.load(f)

    test_x = evaldata['x']
    test_y = evaldata['y']
    
    print("loaded {} instances from : {}".format(test_x.shape[0]))
    print("number of classes : {}".format(len(np.unique(test_y))))
    
    models = ["inception_1", "resnet_1", "rocket_1"]
    explainers = ["lime", "kernelshap"]

    generate_explanations(models, explainers, test_x, savedir)
    