from typing import List, Tuple
import datetime
import argparse
import logging
import pickle
import sys
import os
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import arff
import torch

from resnet import ResNet50
from cnn_mini import CNNMini

logger = logging.getLogger('InsectSound')
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path: str, 
                 device: str, 
                 classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                 transform=None,
                 seed: int = 42):
        
        self.device = device
        self.transform = transform
        
        self.labels_to_num = {
            'Aedes_female': 0, 
            'Aedes_male': 1, 
            'Fruit_flies': 2,
            'House_flies': 3, 
            'Quinx_female': 4, 
            'Quinx_male': 5,
            'Stigma_female': 6, 
            'Stigma_male': 7, 
            'Tarsalis_female': 8,
            'Tarsalis_male': 9
        }

        self.num_to_labels = {v: k for k, v in self.labels_to_num.items()}
        
        logger.info(f'Loading data from {path}')

        with open(path, 'r') as f:
            data = arff.load(f)
        
        all_x = np.array([s[:-1] for s in data["data"]], dtype=np.float32)
        all_y = np.array([self.labels_to_num[s[-1]] for s in data["data"]], dtype=np.int64)

        if classes is None:
            self.x = torch.tensor(all_x, dtype=torch.float32)
            self.y = torch.tensor(all_y, dtype=torch.long)
        
        else:
            mask = []
            for c in classes:
                mask.extend(np.where(all_y == c)[0].tolist())
            
            self.x = torch.tensor(all_x[mask], dtype=torch.float32)
            self.y = torch.tensor(all_y[mask], dtype=torch.long)

        self.classes = torch.unique(self.y).tolist()
    

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx: int):
        
        x = self.x[idx]
        y = self.y[idx]
        return x.to(self.device), y.to(self.device)


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


def evaluate(model: nn.Module, loader: DataLoader):
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, ytrue in loader:
                
            output = model(data)
            ypred = torch.argmax(output, dim=-1)
            
            all_preds.append(ypred)
            all_labels.append(ytrue)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = correct / total

    precision = calculate_precision(all_preds, all_labels)
    
    return accuracy, precision


def train(train_loader : DataLoader, model : nn.Module, config : dict):

    device = config['device']
    epochs = config['epochs']
    lr = config['lr']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model.train()

    logger.info(f'Training model for {epochs} epochs')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Learning rate: {lr}')
    logger.info(f'Using device: {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

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

            accuracy, precision = evaluate(model, train_loader)
            model.train()

            pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: {accuracy:.4f} | prec: {precision:.4f}')

    # save model
    # torch.save(model.state_dict(), f'./models/model_{timestamp}.pth')

def clear_gpu_memory():
    
    torch.cuda.empty_cache()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    logger.info('Cleared GPU memory')


if __name__ == '__main__':
    
    clear_gpu_memory()

    if not os.path.isfile('./README.md'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device

    # classes = [0, 1]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    traindata = InsectDataset('./data/InsectSound_TRAIN.arff', device, classes)
    
    train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)

    model = CNNMini(len(traindata.classes)).to(device)

    config = {
        'device': device,
        'epochs': epochs,
        'lr': lr,
    }

    train(train_loader, model, config)

    logger.info('Evaluating model on test set...')
    
    evaldata = InsectDataset('./data/InsectSound_TEST.arff', device, classes)
    eval_loader = DataLoader(evaldata, batch_size=args.batch_size, shuffle=False)

    accuracy, precision = evaluate(model, eval_loader)

    logger.info(f'Validation accuracy: {accuracy:.4f}')
    logger.info(f'Validation precision: {precision:.4f}')