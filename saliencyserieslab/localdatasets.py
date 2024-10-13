from typing import List
import pickle
import os

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path: str, 
                 device: str, 
                 classes: List[str] = ["all"], 
                 transform=None,
                 seed: int = 42):
        
        self.device = device
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        data_x = data['x']
        data_y = data['y']
        self.class_names = data['labels']
        
        if classes == ["all"]:
            self.x = torch.tensor(data_x, dtype=torch.float32)
            self.y = torch.tensor(data_y, dtype=torch.long)
            self.num_to_labels = {i:c for i, c in enumerate(self.class_names)}
            self.labels_to_num = {c:i for i, c in enumerate(self.class_names)}
            self.n_classes = len(self.class_names)

        if classes != ["all"]:
            empty_x = np.zeros((2500*len(classes), 600), dtype=np.float32)
            empty_y = np.zeros(2500*len(classes), dtype=np.int64)

            for i, c in enumerate(classes):
                
                if c not in self.class_names:
                    raise ValueError(f'Class {c} not in {self.class_names}')

                c = self.class_names.index(c)

                is_class = np.where(data_y == c)[0]

                empty_x[i*2500:(i+1)*2500, :] = data_x[is_class, :]
                empty_y[i*2500:(i+1)*2500] = np.repeat(i, 2500)

            self.x = torch.tensor(empty_x, dtype=torch.float32)
            self.y = torch.tensor(empty_y, dtype=torch.long)

            self.num_to_labels = {i:c for i, c in enumerate(classes)}
            self.labels_to_num = {c:i for i, c in enumerate(classes)}

            self.n_classes = len(classes)
    
    
    @staticmethod
    def get_one_sample(path : str, idx : int, device : str):
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if idx > data['x'].shape[0]:
            raise IndexError('Index out of range')

        x = data['x'][idx]
        y = data['y'][idx]
        label = data['labels'][y]

        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        return x, y, label
    

    @staticmethod
    def get_all(path : str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def __len__(self):
        return self.x.shape[0]


    def __getitem__(self, idx: int):
        
        x = self.x[idx]
        y = self.y[idx]
        return x.to(self.device), y.to(self.device)
