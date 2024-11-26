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


def generate_explanations(
        modelpath : str, 
        test_x : np.ndarray,
        test_y : np.ndarray,
        explainername : str,
        savedir : str,
        dataset : str,
        ):
    
    model = SktimeClassifier()
    model.load_pretrained_model(modelpath)

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

            w = model.model.map_sax_model(test_x[i])[test_y[i]]

            w = np.interp(w, (w.min(), w.max()), (0, 1)).tolist()
            
            with open(full_weight_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(w)

            bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to trained mrsql model. E.g --model ./models/mrseql_ECG200_1')
    args = parser.parse_args()

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    LOGFILE = "./weight_gen_report.csv"

    if not os.path.isfile(LOGFILE):

        with open(LOGFILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "explainer", "dataset", "finished", "runtime", "date", "exception"])

    root = "./weights"
    explainername = "mrseql"
    modelpath = args.model
    dataset = modelpath.split("/")[-1].split("_")[1]

    if not os.path.isdir(root):
        os.mkdir(root)    

    with open("./data/{}.pkl".format(dataset), 'rb') as f:
        data = pickle.load(f)

    test_x, test_y = data[2], data[3]

    starttime = time.time()

    generate_explanations(
        modelpath=modelpath, 
        test_x=test_x,
        test_y=test_y,
        explainername=explainername,
        savedir=root,
        dataset=dataset,
        )
    
    endtime = time.time()

    runtime = (endtime - starttime) / 60

    with open(LOGFILE, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([modelpath, explainername, dataset, True, runtime, datetime.datetime.now(), None])
