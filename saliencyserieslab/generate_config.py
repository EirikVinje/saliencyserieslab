import json
import os

import optuna

def generate_config(configpath : str = None):

    config = {
        "inception" : {},
        "rocket" : {},
        "resnet" : {},
        }

    config["inception"]["bottleneck_size"] = 32
    config["inception"]["random_state"] = 42
    config["inception"]["kernel_size"] = 5
    config["inception"]["batch_size"] = 64
    config["inception"]["n_filters"] = 64
    config["inception"]["n_epochs"] = 30
    config["inception"]["verbose"] = 1
    config["inception"]["depth"] = 5

    config["rocket"]["max_dilations_per_kernel"] = 8
    config["rocket"]["rocket_transform"] = "rocket"
    config["rocket"]["n_features_per_kernel"] = 8
    config["rocket"]["num_kernels"] = 5000
    config["rocket"]["random_state"] = 42

    config["resnet"]["random_state"] = 42
    config["resnet"]["batch_size"] = 64
    config["resnet"]["n_epochs"] = 30    
    config["resnet"]["verbose"] = 1

    if configpath is not None:
        with open(configpath, 'w') as f:
            json.dump(config, f)
    
    else:
        return config


def generate_hp_config(modelname : str, trial : optuna.trial.Trial):
    
    config = {
        "inception" : {},
        "rocket" : {},
        "resnet" : {},
        }

    if modelname == 'inception':

        config["inception"]["kernel_size"] = trial.suggest_categorical("kernel_size", [10, 20, 30, 40, 50, 60])
        config["inception"]["bottleneck_size"] = trial.suggest_categorical("bottleneck_size", [16, 32, 48])
        config["inception"]["n_filters"] = trial.suggest_categorical("n_filters", [16, 32, 48])
        config["inception"]["depth"] = trial.suggest_categorical("depth", [3, 6, 9])
        
        config["inception"]["random_state"] = 42
        config["inception"]["batch_size"] = 64
        config["inception"]["n_epochs"] = 30
        config["inception"]["verbose"] = 1

    elif modelname == 'rocket':

        config["rocket"]["rocket_transform"] = trial.suggest_categorical("rocket_transform", ["rocket", "minirocket", "multirocket"])
        config["rocket"]["max_dilations_per_kernel"] = trial.suggest_categorical("max_dilations_per_kernel", [16, 32, 48])
        config["rocket"]["n_features_per_kernel"] = trial.suggest_categorical("n_features_per_kernel", [2, 4, 6, 8, 10])
        config["rocket"]["num_kernels"] = 2500
        config["rocket"]["random_state"] = 42

    elif modelname == 'resnet':

        config["resnet"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
        config["resnet"]["random_state"] = 42
        config["resnet"]["n_epochs"] = 30    
        config["resnet"]["verbose"] = 1

    return config