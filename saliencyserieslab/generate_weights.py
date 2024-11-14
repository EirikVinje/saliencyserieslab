import argparse
import datetime
import pickle
import time
import csv
import os

from tqdm import tqdm
import numpy as np


from saliencyserieslab.explainers.load_explainer import load_explainer
from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.load_data import UcrDataset


def generate_explanations(
        modelpath : str, 
        explainername : str, 
        test_x : np.ndarray,
        test_y : np.ndarray,
        savedir : str,
        dataset
        ):
    
    model = SktimeClassifier()
    model.load_pretrained_model(modelpath)

    explainer = load_explainer(
        model, 
        explainername,
        test_x,
        )

    W = []

    modelname = modelpath.split("/")[-1].split("_")[0]

    weight_path = "{}_{}_{}.csv".format(modelname, explainername, dataset)
    full_weight_path = os.path.join(savedir, weight_path)

    if os.path.isfile(full_weight_path):
        print("removing old weight file {}".format(full_weight_path))
        os.remove(full_weight_path)
    
    with open(full_weight_path, 'w') as f:
        writer = csv.writer(f)


    with tqdm(total=test_x.shape[0], desc="generating explanations ({} - {} - {})".format(modelname, explainername, dataset)) as bar:

        for i in range(test_x.shape[0]):

            w = explainer.explain_instance(test_x[i], test_y[i])
            
            with open(os.path.join(savedir, weight_path), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(w)

            bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to trained model. E.g --model ./models/mrseql_ECG200_1')
    parser.add_argument('--explainer', type=str, help='name of explainer. E.g --explainer kernelshap')
    parser.add_argument('--testsize', type=int, default=None, help='size of test set to use. E.g --testsize 1000')
    args = parser.parse_args()

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    LOGFILE = "./weight_gen_report.csv"

    if not os.path.isfile(LOGFILE):

        with open(LOGFILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "explainer", "dataset", "finished", "runtime", "date", "exception"])


    root = "./weights"
    explainername = args.explainer
    modelname = args.model
    dataset = modelname.split("/")[-1].split("_")[1]
    testsize = args.testsize


    if not os.path.isdir(root):
        os.mkdir(root)    

    ucr = UcrDataset(
        name=dataset,
        float_dtype=32,
        scale=False,
    )

    test_x, test_y = ucr.load_split("test", testsize)

    starttime = time.time()

    generate_explanations(
        modelname, 
        explainername,
        test_x,
        test_y,
        root,
        dataset,
        )
    
    endtime = time.time()

    runtime = (endtime - starttime) / 60

    with open(LOGFILE, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([modelname, explainername, dataset, True, runtime, datetime.datetime.now(), None])