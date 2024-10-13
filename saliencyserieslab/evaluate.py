from typing import List
import argparse
import datetime
import logging
import json
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch

from localdatasets import InsectDataset
from modelutils import load_state_dict
from modelutils import model_selection

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger('src')

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
        
        macro_precision = torch.tensor(precisions).mean().item()
    
    return macro_precision


def calculate_accuracy(predictions, labels):

    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    return accuracy


def calculate_metrics(model: nn.Module, loader: DataLoader):
    
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


if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/modelconfig.json', help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to model file, e.g ./models/resnet_20221017_092600.pth')
    args = parser.parse_args()

    config_path = args.config
    state_dict_path = args.model

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    logger.info(f'Loaded config from : {config_path}')

    evaldata = InsectDataset(config['testpath'], config['device'], config['classes'])
    eval_loader = DataLoader(evaldata, batch_size=config['batch_size'], shuffle=False)
    logger.info('Loaded eval data from : {}'.format(config['testpath']))

    model = model_selection(config, n_classes=evaldata.n_classes)
    state_dict = load_state_dict(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(config['device'])
    model.eval()
    logger.info('Loaded trained model from : {}'.format(state_dict_path))
    
    accuracy, precision = calculate_metrics(model, eval_loader)

    logger.info(f'Accuracy : {accuracy}')
    logger.info(f'Precision : {precision}')
    
    resultpath = './results/results_{}.json'.format(timestamp)

    logger.info('Saving results to : {}'.format(resultpath))

    with open(resultpath, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'precision': precision,
        }, f)