

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification
from scipy import signal
import numpy as np


class UcrDataset:
    def __init__(
            self, 
            name : str, 
            n_dims : int = None,
            float_dtype : int = 32,
            scale : bool = True,
            random_seed : int = 42,
            ):
        
        self.name = name
        self.n_dims = n_dims
        self.float_dtype = float_dtype
        self.scale = scale
        self.random_seed = random_seed
        

    def reduce_dims(self, x, n, method='decimate'):
            
            if len(x) < n:
                raise ValueError("Output length must be smaller than input length")

            if method == 'decimate':
                
                factor = len(x) // n
                if factor > 1:
                    reduced_ts = signal.decimate(x, factor, n=int(np.ceil(len(x)/n)))
                else:
                    reduced_ts = x
            
            elif method == 'peaks':
            
                peaks, _ = signal.find_peaks(x)
                if len(peaks) > n:
            
                    peak_values = x[peaks]
                    sorted_indices = np.argsort(peak_values)[-n:]
                    selected_peaks = peaks[sorted_indices]
                else:
                    selected_peaks = peaks

                x_original = np.linspace(0, len(x)-1, len(x))
                x_new = np.linspace(0, len(x)-1, n)
                reduced_ts = np.interp(x_new, x_original[selected_peaks], x[selected_peaks])
            else:
            
                splits = np.array_split(x, n)

                if method == 'mean':
                    reduced_ts = np.array([chunk.mean() for chunk in splits])
                elif method == 'max':
                    reduced_ts = np.array([chunk.max() for chunk in splits])
                elif method == 'min':
                    reduced_ts = np.array([chunk.min() for chunk in splits])
                elif method == 'median':
                    reduced_ts = np.array([np.median(chunk) for chunk in splits])
                else:
                    raise ValueError(f"Unknown method: {method}")

            return reduced_ts


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

        if self.n_dims is not None:
            x = [self.reduce_dims(x[i], self.n_dims) for i in range(x.shape[0])]
            x = np.vstack(x)

        if self.float_dtype == 16:
            x = x.astype(np.float16)
        elif self.float_dtype == 32:
            x = x.astype(np.float32)
        elif self.float_dtype == 64:
            x = x.astype(np.float64)
        
        # y = y.astype(np.int8)

        if size is not None and size < x.shape[0]:
            _, x, _, y = train_test_split(x, y, test_size=float(size/x.shape[0]), random_state=self.random_seed)

        return x, y