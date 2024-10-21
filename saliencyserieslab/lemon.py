from typing import List, Callable
import multiprocessing
import logging

from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
import numpy as np

# logger = logging.getLogger('src')
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


class LemonExplainer:
    def __init__(self, 
                 model_fn : Callable, 
                 perturbation_ratio : float=0.4, 
                 adjacency_prob : float=0.9, 
                 num_samples : int=5000, 
                 segment_size : int=1, 
                 sigma : float=0.1):
    
        self.perturbation_ratio = perturbation_ratio
        self.adjacency_prob = adjacency_prob
        self.segment_size = segment_size
        self.num_samples = num_samples
        self.model_fn = model_fn
        self.sigma = sigma

        # logger.info(f"Initialized explainer")
        # logger.info(f"Adjacency probability: {self.adjacency_prob}")
        # logger.info(f"Perturbation ratio: {self.perturbation_ratio}")
        # logger.info(f"Number of samples: {self.num_samples}")
        # logger.info(f"Segment size: {self.segment_size}")
        # logger.info(f"Sigma: {self.sigma}")


    def _distance(self, original, perturbed, method='euclidean'):
        """
        Calculate the distance/similarity between the original time series and a perturbed sample.
        
        :param original: The original time series (1D numpy array)
        :param perturbed: The perturbed time series (1D numpy array)
        :param method: The distance metric to use ('dtw', 'euclidean', or 'correlation')
        :param sigma: Parameter for the RBF kernel to convert distance to similarity
        :return: A similarity score between 0 and 1
        """
        if method == 'dtw':
            pass
            # distance = dtw.distance(original, perturbed)
        elif method == 'euclidean':
            distance = euclidean(original, perturbed)
        elif method == 'correlation':
            correlation = np.corrcoef(original, perturbed)[0, 1]
            distance = 1 - abs(correlation)  # Convert correlation to distance
        else:
            raise ValueError("Unsupported distance method. Choose 'dtw', 'euclidean', or 'correlation'.")
        
        # Convert distance to similarity using RBF kernel
        similarity = np.exp(-(distance ** 2) / (2 * self.sigma ** 2))
        
        return similarity


    def _perturb_data(self, x : np.ndarray):
        """
        Create perturbed samples of a time series with adjacent segments more likely to be perturbed together.
        
        :param timeseries: Original time series data (1D numpy array)
        :param num_samples: Number of perturbed samples to generate
        :param segment_size: Size of each segment
        :param perturbation_ratio: Ratio of segments to perturb
        :param adjacency_prob: Probability of perturbing an adjacent segment
        :return: Perturbed samples and their binary representations
        """
    
        assert len(x.shape) == 1, "Input must be a 1D numpy array"
        
        # Ensure the timeseries length is divisible by segment_size
        padded_length = ((x.shape[0] - 1) // self.segment_size + 1) * self.segment_size
        padded_timeseries = np.pad(x, (0, padded_length - x.shape[0]), mode='constant', constant_values=0)
        
        # Convert to float to allow for noise addition
        padded_timeseries = padded_timeseries.astype(float)
        
        num_segments = padded_length // self.segment_size
        num_perturb = int(num_segments * self.perturbation_ratio)
        
        perturbed_samples = []
        binary_representations = []
        
        for _ in range(self.num_samples):
            # Initialize perturbation mask
            perturb_mask = np.zeros(num_segments, dtype=bool)
            
            # Start with a random segment
            current_segment = np.random.randint(0, num_segments)
            perturb_mask[current_segment] = True
            
            # Perturb adjacent segments with higher probability
            while np.sum(perturb_mask) < num_perturb:
                if np.random.random() < self.adjacency_prob:
                    # Perturb adjacent segment
                    direction = np.random.choice([-1, 1])  # Left or right
                    next_segment = (current_segment + direction) % num_segments
                else:
                    # Jump to a random unperturbed segment
                    unperturbed = np.where(~perturb_mask)[0]
                    next_segment = np.random.choice(unperturbed)
                
                perturb_mask[next_segment] = True
                current_segment = next_segment
            
            # Create a copy of the original series
            perturbed = np.copy(padded_timeseries)
            binary_rep = np.ones(num_segments)
            
            for idx in np.where(perturb_mask)[0]:
                start = idx * self.segment_size
                end = start + self.segment_size
                
                # Randomly choose a perturbation method
                method = np.random.choice(['zero', 'noise', 'shuffle'])
                
                if method == 'zero':
                    perturbed[start:end] = 0
                elif method == 'noise':
                    perturbed[start:end] += np.random.normal(0, np.std(x), self.segment_size)
                elif method == 'shuffle':
                    np.random.shuffle(perturbed[start:end])
                
                binary_rep[idx] = 0
            
            perturbed_samples.append(perturbed[:x.shape[0]])
            binary_representations.append(binary_rep)
        
        return np.array(perturbed_samples), np.array(binary_representations)


    def explain(self, x : np.ndarray):
        
        # logger.info(f"Producing data perturbations...")
        x_perturb, binary_rep = self._perturb_data(x)

        # logger.info(f"Calculating perturbation weights...")
        weights = [self._distance(x, x_perturb[i]) for i in range(len(x_perturb))]
        
        # logger.info(f"Predicting perturbed data...")
        perturb_predictions = self.model_fn(x_perturb)
        
        # logger.info(f"Fitting logistic regression model...")
        logreg = LogisticRegression(solver='lbfgs', n_jobs=multiprocessing.cpu_count()-2, max_iter=1000)
        logreg.fit(binary_rep, perturb_predictions, sample_weight=weights)

        w = logreg.coef_[0].reshape(-1)
        w = np.interp(w, (w.min(), w.max()), (0, 1))
        w = np.repeat(w, x.shape[0] // w.shape[0])
                
        return w