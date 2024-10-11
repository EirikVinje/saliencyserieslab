from typing import List

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import arff

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
