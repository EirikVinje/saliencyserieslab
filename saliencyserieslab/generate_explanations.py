from typing import List, Dict
import subprocess
import argparse
import pickle
import json
import os

from tqdm import tqdm
import numpy as np

from load_sktime_classifier import SktimeClassifier
from load_explainer import Explainer


def generate_explanations(models : List, explainers : List, test_x : np.ndarray, run_id : int):

    savedir = "./explanations/exp_{}".format(run_id)

    if os.path.isdir(savedir):
        raise RuntimeError("run id {} already exists. Change run id or delete the directory".format(run_id))

    os.makedir(savedir)

    with tqdm(total=len(models) * len(explainers)) as bar:

        for modelname in models:
            for explainer in explainers:
                bar.set_description("generating weights for pair : ({}, {})".format(modelname, explainer))

                command = ["python", "-m", "saliencyserieslab.explain", "--model", modelname, "--explainer", explainer, "--id", run_id, "--savedir", savedir]
                subprocess.run(command)



if __name__ == "__main__":

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test.pkl', help='Path to eval data')
    parser.add_argument('--id', type=str, help='model id')
    args = parser.parse_args()

    testpath = args.testpath
    RUN_ID = args.id

    models = ["inception", "resnet", "rocket"]
    explainers = ["lime", "kernelshap"]

    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)

    generate_explanations(models, explainers, evaldata["x"])