import gc

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch

from cnn_mini import CNNMini
from resnet import ResNet50


def model_selection(config : dict):

    if config['modelname'] == 'resnet':
        model = ResNet50(len(config['classes'])).to(config['device'])
    elif config['modelname'] == 'cnnmini':
        model = CNNMini(len(config['classes'])).to(config['device'])
    else:
        raise ValueError('Invalid model name')
    
    return model


def save_state_dict(model_state_dict: dict, path: str):
    torch.save(model_state_dict, path)


def load_state_dict(path: str):
    return torch.load(path)


def calculate_precision(predictions, labels):
    with torch.no_grad():
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        precisions = []
        for class_id in range(torch.unique(labels).size(0)):
            true_positives = ((predictions == class_id) & (labels == class_id)).sum().float()
            predicted_positives = (predictions == class_id).sum().float()
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
            precisions.append(precision)
        
        macro_precision = torch.tensor(precisions).mean()
    
    return macro_precision


def calculate_accuracy(predictions, labels):

    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    return accuracy


def evaluate(model: nn.Module, loader: DataLoader):
    
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for data, ytrue in loader:
                
            output = model(data)
            ypred = torch.argmax(output, dim=-1)
            
            all_preds.append(ypred)
            all_labels.append(ytrue)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = calculate_accuracy(all_preds, all_labels)
    precision = calculate_precision(all_preds, all_labels)
    
    return accuracy, precision


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
    plt.savefig(f'./log/results_{timestamp}.png')


def smooth_data(data, window_size):
    
    if window_size % 2 == 0:
        window_size += 1
    
    data = np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
