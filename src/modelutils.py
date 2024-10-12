import gc

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch

from cnn_mini import CNNMini
from resnet import ResNet50


def model_selection(config : dict, n_classes : int):

    if config['modelname'] == 'resnet':
        model = ResNet50(n_classes).to(config['device'])
    elif config['modelname'] == 'cnnmini':
        model = CNNMini(n_classes).to(config['device'])
    else:
        raise ValueError('Invalid model name')
    
    return model


def save_state_dict(model_state_dict: dict, path: str):
    torch.save(model_state_dict, path)


def load_state_dict(path: str):
    return torch.load(path)


def clear_gpu_memory():
    
    torch.cuda.empty_cache()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def plot_results(accuracy_log, precision_log, loss_log, timestamp : str):
    
    accuracy_log = smooth_data(accuracy_log, 100)
    precision_log = smooth_data(precision_log, 100)
    loss_log = smooth_data(loss_log, 100)

    plt.plot(accuracy_log, label='Accuracy')
    plt.plot(precision_log, label='Precision')
    plt.plot(loss_log, label='Loss')
    plt.legend()
    plt.savefig(f'./plots/plot_results_{timestamp}.png')


def smooth_data(data, window_size):
    
    if window_size % 2 == 0:
        window_size += 1
    
    data = np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
