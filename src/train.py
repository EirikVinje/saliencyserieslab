import datetime
import argparse
import logging
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
from modelutils import evaluate


logger = logging.getLogger('InsectSound')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.update(1)

            lr_scheduler.step()

            if log:

                accuracy, precision = evaluate(model, train_loader)
                model.train()

                pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: {accuracy:.4f} | prec: {precision:.4f}')

                accuracy = float(accuracy)
                precision = float(precision)
                loss = float(loss)
                
                accuracy_log.append(accuracy)
                precision_log.append(precision)
                loss_log.append(loss)
            
            else:
                pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: - | prec: -')
            
    if log:
        return model.state_dict(), accuracy_log, precision_log, loss_log

    else:
        return model.state_dict()


if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f'Timestamp: {timestamp}')

    clear_gpu_memory()
    logger.info('Cleared GPU memory')

    if not os.path.isfile('./README.md'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='cnnmini')
    args = parser.parse_args()

    config = {
        'device': args.device,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'modelname': args.model,
        'classes': ["all"],
        'trainpath' : './data/InsectSound_TRAIN.arff',
        'testpath' : './data/InsectSound_TEST.arff'
    }

    config["classes"] = ['Aedes_female',
                        'Aedes_male',
                        'Fruit_flies',
                        'House_flies',]

    logger.info(f'Loading train from : {config["trainpath"]}')
    traindata = InsectDataset(config['trainpath'], config['device'], config['classes'])
    train_loader = DataLoader(traindata, batch_size=config['batch_size'], shuffle=True)

    model = model_selection(config)

    logger.info(f"Using model: {config['modelname']}")

    modelstate, accuracy_log, precision_log, loss_log = train(train_loader, model, config, timestamp)
    
    assert False

    #! TODO : save model and load model to explain.py
    #! TODO : Finish LIME

    modelsavepath = f'./models/{config["modelname"]}_{timestamp}.pth'
    
    save_state_dict(modelstate, modelsavepath)
    logger.info(f'Saved model to {modelsavepath}')

    logger.info('Evaluating model on test set...')
    logger.info(f'Loading test from : {config["testpath"]}')
    
    evaldata = InsectDataset(config['testpath'], config['device'], config['classes'])
    eval_loader = DataLoader(evaldata, batch_size=config['batch_size'], shuffle=False)

    accuracy, precision = evaluate(model, eval_loader)

    logger.info(f'Validation accuracy: {accuracy:.4f}')
    logger.info(f'Validation precision: {precision:.4f}')

    plot_results(accuracy_log, precision_log, loss_log, timestamp)