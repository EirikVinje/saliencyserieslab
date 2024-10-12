from typing import List

from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io.arff import loadarff
import torch


class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path: str, 
                 device: str, 
                 classes: List[str] = ["all"], 
                 transform=None,
                 seed: int = 42):
        
        self.device = device
        self.transform = transform
        
        self.class_names = [
                'Aedes_female',
                'Aedes_male',
                'Fruit_flies',
                'House_flies',
                'Quinx_female',
                'Quinx_male',
                'Stigma_female',
                'Stigma_male',
                'Tarsalis_female',
                'Tarsalis_male'
                ]

        data = loadarff(open(path, 'r'))

        data = [list(d) for d in data[0]]

        data_x = [d[:-1] for d in data]
        data_x = np.array(data_x, dtype=np.float32)

        data_y = [d[-1] for d in data]
        data_y = list(map(lambda x: x.decode('utf-8'), data_y))
        data_y = np.array(data_y, dtype=str)
        
        if classes == ["all"]:
            
            unique = np.unique(data_y)
            self.labels_to_num = {str(c):i for i, c in enumerate(unique)}
            self.num_to_labels = {i:str(c) for i, c in enumerate(unique)}
        
            data_y = [self.labels_to_num[c] for c in data_y]
            data_y = np.array(data_y, dtype=np.int64)
            
            self.x = torch.tensor(data_x, dtype=torch.float32)
            self.y = torch.tensor(data_y, dtype=torch.long)

            self.n_classes = unique.shape[0]

        else:

            empty_x = np.zeros((2500*len(classes), 600), dtype=np.float32)
            empty_y = np.zeros(2500*len(classes), dtype=np.int64)

            for i, c in enumerate(classes):
                
                if c not in self.class_names:
                    raise ValueError(f'Class {c} not in {self.class_names}')

                is_class = np.where(data_y == c)[0]

                empty_x[i*2500:(i+1)*2500, :] = data_x[is_class, :]
                empty_y[i*2500:(i+1)*2500] = np.repeat(i, 2500)

            self.x = torch.tensor(empty_x, dtype=torch.float32)
            self.y = torch.tensor(empty_y, dtype=torch.long)

            self.num_to_labels = {i:c for i, c in enumerate(classes)}
            self.labels_to_num = {c:i for i, c in enumerate(classes)}

            self.n_classes = len(classes)


    @staticmethod
    def get_class_names():
        
        classes = [
            'Aedes_female',
            'Aedes_male',
            'Fruit_flies',
            'House_flies',
            'Quinx_female',
            'Quinx_male',
            'Stigma_female',
            'Stigma_male',
            'Tarsalis_female',
            'Tarsalis_male'
        ]

        return classes

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx: int):
        
        x = self.x[idx]
        y = self.y[idx]
        return x.to(self.device), y.to(self.device)
