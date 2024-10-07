import datetime
import argparse
import logging
import pickle
import os

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn
import torch

from resnet import ResNet50
from cnn_mini import CNNMini

class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, path, device):
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.x = data['x']
        self.y = data['y']

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx].to(device), self.y[idx].to(device)


def calculate_precision(predictions, labels, num_classes):
    with torch.no_grad():
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        precisions = []
        for class_id in range(num_classes):
            true_positives = ((predictions == class_id) & (labels == class_id)).sum().float()
            predicted_positives = (predictions == class_id).sum().float()
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
            precisions.append(precision)
        
        macro_precision = torch.tensor(precisions).mean()
    
    return macro_precision



def evaluate(model: nn.Module, loader: DataLoader, device: str):
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, ytrue in loader:
            data = data.to(device)
            ytrue = ytrue.to(device).long()
            
            output = model(data)
            ypred = torch.argmax(output, dim=-1)
            
            all_preds.append(ypred)
            all_labels.append(ytrue)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    acc = correct / total

    # precision = calculate_precision(all_preds, all_labels, 2)

    return acc


def train(train_loader : DataLoader,
          model : nn.Module, 
          device : str, 
          epochs : int,
          lr : float):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    with tqdm(total=epochs * len(train_loader)) as pbar:
        
        pbar.set_description(f'epoch 0:{epochs} | loss: - | acc: -')
        
        for i in range(epochs):
            
            for batch_idx, (data, ytrue) in enumerate(train_loader):
                
                optimizer.zero_grad()
                
                output = model(data)

                loss = criterion(output, ytrue)
                loss.backward()
                
                optimizer.step()

                pbar.update(1)
            
            lr_scheduler.step()

            trainacc = evaluate(model, train_loader, device)

            pbar.set_description(f'epoch {i+1}:{epochs} | loss: {loss:.4f} | acc: {trainacc:.4f}')

    # save model
    # torch.save(model.state_dict(), f'./models/model_{timestamp}.pth')


if __name__ == '__main__':
    
    if not os.path.isfile('./README.md'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--x_path', type=str, required=True)
    # parser.add_argument('--y_path', type=str, required=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device

    train_path = './data/TRAIN_2class.pkl'
    
    dataset = InsectDataset(train_path, device)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CNNMini(6).to(device)

    train(train_loader, model, device, epochs, lr)