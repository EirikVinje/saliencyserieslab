import datetime
import argparse
import logging
import json
import sys
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

from localdatasets import InsectDataset
from modelutils import clear_gpu_memory
from modelutils import save_state_dict
from modelutils import model_selection
from modelutils import plot_results
from evaluate import calculate_metrics

# set torch seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger('src')

def train(train_loader : DataLoader, model : nn.Module, config : dict, timestamp : str, log: bool = True):

    batch_size = config['batch_size']
    classes = config['classes']
    device = config['device']
    epochs = config['epochs']
    lr = config['lr']

    model.train()

    logger.info(f'Classes : {classes}')
    logger.info(f'Epochs : {epochs}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Learning rate: {lr}')
    logger.info(f'Using device: {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    accuracy_log = []
    precision_log = []
    loss_log = []

    with tqdm(total=epochs * len(train_loader)) as pbar:
        
        pbar.set_description(f'epoch 0:{epochs} | loss: - | acc: - | prec: -')
        
        for i in range(epochs):
            
            for batch_idx, (data, ytrue) in enumerate(train_loader):
                
                optimizer.zero_grad()
                
                output = model(data)
                
                loss = criterion(output, ytrue)
                loss.backward()
                
                optimizer.step()

                pbar.update(1)

            lr_scheduler.step()

            if log:
                accuracy, precision = calculate_metrics(model, train_loader)
                model.train()

                pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: {accuracy:.4f} | prec: {precision:.4f}')

                accuracy_log.append(float(accuracy))
                precision_log.append(float(precision))
                loss_log.append(float(loss))
            
            else:
                pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: - | prec: -')
            
    if log:
        return model.state_dict(), accuracy_log, precision_log, loss_log

    else:
        return model.state_dict()


if __name__ == '__main__':

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f'Timestamp: {timestamp}')

    clear_gpu_memory()
    logger.info('Cleared GPU memory')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='./modelconfigs/modelconfig.json', help='Path to config file')
    parser.add_argument('--log', type=argparse.BooleanOptionalAction, default=False, help='Whether to log results')
    args = parser.parse_args()

    do_log = args.log

    with open(args.config, 'r') as json_file:
        config = json.load(json_file)

    logger.info(f'Loading train from : {config["trainpath"]}')
    traindata = InsectDataset(config['trainpath'], config['device'], config['classes'])
    train_loader = DataLoader(traindata, batch_size=config['batch_size'], shuffle=True)

    model = model_selection(config, traindata.n_classes)

    model = model.to(config['device'])

    logger.info(f"Using model: {config['modelname']}")

    modelstate, accuracy_log, precision_log, loss_log = train(train_loader, model, config, timestamp)
    plot_results(accuracy_log, precision_log, loss_log, timestamp)
    

    modelsavepath = f'./models/{config["modelname"]}_{timestamp}.pth'
    save_state_dict(modelstate, modelsavepath)
    logger.info(f'Model saved to : {modelsavepath}')
