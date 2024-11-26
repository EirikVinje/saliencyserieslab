

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification
from scipy import signal
import numpy as np


class UcrDataset:
    def __init__(
            self, 
            name : str, 
            float_dtype : int = 32,
            scale : bool = True,
            random_seed : int = 42,
            ):
        
        self.name = name
        self.float_dtype = float_dtype
        self.scale = scale
        self.random_seed = random_seed

    def load_split(self, split="train", size: int = None):

        train = load_classification(self.name, split)
        
        unique_classes = np.unique(train[1]).tolist()
        unique_classes = [int(c) for c in unique_classes]
        unique_classes.sort()
        
        x = train[0].squeeze()
        y = np.array([unique_classes.index(int(c)) for c in train[1]])

        if self.scale:
            scaler = StandardScaler()
            x = scaler.fit_transform(x, y)

        if self.float_dtype == 16:
            x = x.astype(np.float16)
        elif self.float_dtype == 32:
            x = x.astype(np.float32)
        elif self.float_dtype == 64:
            x = x.astype(np.float64)
        
        return x, y
