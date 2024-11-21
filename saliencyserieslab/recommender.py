from typing import List, Any
import pickle
import json
import csv
import os

from numba import njit
from tqdm import tqdm
import pandas as pd
import numpy as np

from saliencyserieslab.generate_testset import PerturbedDataGenerator
from saliencyserieslab.classifier import SktimeClassifier


def load_explanations(
        modelname : str,
        explainername : str,
        dataset : str,
        rootdir : str = "./weights",
        ):

    weight_path = "{}_{}_{}.csv".format(modelname, explainername, dataset)
    full_weight_path = os.path.join(rootdir, weight_path)

    if not os.path.isfile(full_weight_path):
        raise RuntimeError("weight file {} does not exist".format(full_weight_path))

    data = []
    with open(full_weight_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    
    data = np.array(data, dtype=np.float32)
    return data


def compute_eauc(
        model : Any, 
        W : np.ndarray, 
        test_x : np.ndarray, 
        test_y : np.ndarray, 
        perturb_method : str, 
        ):

    accuracy = []
                                
    top_k_generator = PerturbedDataGenerator()
    
    k = np.arange(0.0, 1.01, 0.1)

    for i, ki in enumerate(k):    
        
        perturbed_x_top_k = top_k_generator(
            method=perturb_method,
            X=test_x, 
            k=ki,
            W=W, 
        )
        
        acc = model.evaluate(perturbed_x_top_k, test_y)
        accuracy.append(acc)
    
    return accuracy


def recommender(
        models : List, 
        explainers : List,
        datasets : List,
        perturb_method : str,
        progress_bar : bool = True,
        ):

    if perturb_method not in ["local_mean", "global_mean", "local_gaussian", "global_gaussian"]:
        raise ValueError("perturb_method must be in ['local_mean', 'global_mean', 'local_gaussian', 'global_gaussian']")
    
    record_euc = {}

    with tqdm(total=len(models) * len(explainers), disable=not progress_bar) as bar:
    
        for modelpath in models:

            model = SktimeClassifier()
            model.load_pretrained_model(modelpath)

            dataset_name = modelpath.split("/")[-1].split("_")[1]
            model_name = modelpath.split("/")[-1].split("_")[0]
            
            if model_name not in record_euc.keys():
                record_euc[model_name] = {}

            test_x, test_y = datasets[dataset_name]

            if dataset_name not in record_euc[model_name].keys():
                record_euc[model_name][dataset_name] = {}

            for explainer_name in explainers:

                bar.set_description("EAUC : ({} - {} - {})".format(model_name, explainer_name, dataset_name))
                
                if explainer_name == "mrseql":
                    W = load_explanations("mrseql", explainer_name, dataset_name)    

                else:
                    W = load_explanations(model_name, explainer_name, dataset_name)    
                
                accuracy = compute_eauc(
                    perturb_method=perturb_method,
                    test_x=test_x, 
                    test_y=test_y, 
                    model=model, 
                    W=W,
                    )

                record_euc[model_name][dataset_name][explainer_name] = accuracy

                bar.update(1)

    return record_euc                    


if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    models = [
        "./models2/rocket_SwedishLeaf_1",
        "./models2/rocket_ECG200_1",
        "./models2/rocket_Plane_1",
        "./models2/mrseql_SwedishLeaf_1",
        "./models2/mrseql_ECG200_1",
        "./models2/mrseql_Plane_1",
        "./models2/resnet_SwedishLeaf_1",
        "./models2/resnet_ECG200_1",
        "./models2/resnet_Plane_1",
    ]

    explainers = [
        "shapley",
        "kernelshap",
        "leftist_shap",
        "leftist_lime",
        "lime",
        "mrseql",
    ]

    datasets = {}
    for dataset in ["SwedishLeaf", "ECG200", "Plane"]:

        with open("./data/{}.pkl".format(dataset), 'rb') as f:
            ucr = pickle.load(f)
        test_x, test_y = ucr[2], ucr[3]
        datasets[dataset] = (test_x, test_y)

    lg_eauc = recommender(models, explainers, datasets, "local_gaussian")
    with open("./results/EAUC_local_gaussian.json", 'w') as f:
        json.dump(lg_eauc, f, indent=4)

    gg_eauc = recommender(models, explainers, datasets, "global_gaussian")
    with open("./results/EAUC_global_gaussian.json", 'w') as f:
        json.dump(gg_eauc, f, indent=4)

    lm_eauc = recommender(models, explainers, datasets, "local_mean")
    with open("./results/EAUC_local_mean.json", 'w') as f:
        json.dump(lm_eauc, f, indent=4)
    
    gm_eauc = recommender(models, explainers, datasets, "global_mean")
    with open("./results/EAUC_global_mean.json", 'w') as f:
        json.dump(gm_eauc, f, indent=4)

    eauc = {}
    eauc["global_gaussian"] = gg_eauc
    eauc["local_gaussian"] = lg_eauc
    eauc["global_mean"] = gm_eauc
    eauc["local_mean"] = lm_eauc

    with open("./results/EAUC.json", 'w') as f:
        json.dump(eauc, f, indent=4)