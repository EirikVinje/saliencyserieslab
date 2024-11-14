from typing import List
import pickle
import json
import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from saliencyserieslab.generate_testset import PerturbedDataGenerator
from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.load_data import UcrDataset


def load_explanations(
        modelname : str,
        explainername : str,
        dataset : str,
        rootdir : str = "./weights",
        ):

    weight_path = "{}_{}_{}.csv".format(modelname, explainername, dataset)
    full_weight_path = os.path.join(rootdir, weight_path)

    return pd.read_csv(full_weight_path).to_numpy()


def recommender_3(
        models : List, 
        explainers : List, 
        savepath : str,
        progress_bar : bool = True,
        ):

    record_euc = {"local_mean" : {}, "global_mean" : {}}

    with tqdm(total=len(models) * len(explainers)) as bar:
    
        for modelpath in models:

            model = SktimeClassifier()
            model.load_pretrained_model(modelpath)

            dataset_name = modelpath.split("/")[-1].split("_")[1]
            model_name = modelpath.split("/")[-1].split("_")[0]

            ucr = UcrDataset(
                name=dataset_name,
                float_dtype=32,
                scale=False,
            )

            test_x, test_y = ucr.load_split("test")

            for explainer_name in explainers:
                
                bar.set_description("calculating AUC for : ({} - {} - {})".format(model_name, explainer_name, dataset_name))

                W = load_explanations(model_name, explainer_name, dataset_name)
                
                for method in record_euc.keys():
                    
                    if model_name not in record_euc[method]:
                        record_euc[method][model_name] = {}

                    accuracy = []
                    k = np.arange(0.0, 1.01, 0.1)
                    for i, ki in enumerate(k):
                        
                        top_k_generator = PerturbedDataGenerator()
                        
                        perturbed_x_top_k = top_k_generator(
                            method=method,
                            X=test_x, 
                            k=ki,
                            W=W, 
                        )

                        acc = model.evaluate(perturbed_x_top_k, test_y)
                        accuracy.append(acc)
                    
                    eauc = np.trapz(x=k, y=accuracy) 
                    record_euc[method][model_name][explainer_name] = eauc

    with open(savepath, 'wb') as f:
        pickle.dump(record_euc, f)
                        

if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    models = [
        "./models/mrseql_SwedishLeaf_1",
        "./models/rocket_SwedishLeaf_1",
        "./models/weasel_SwedishLeaf_1",
        "./models/rocket_ECG200_1",
        "./models/mrseql_ECG200_1",
        "./models/weasel_ECG200_1",
        "./models/rocket_Plane_1",
        "./models/mrseql_Plane_1",
        "./models/weasel_Plane_1",
    ]

    explainers = [
        "shapley",
        "kernelshap",
        "leftist_shap",
        "leftist_lime",
        "lime",
    ]