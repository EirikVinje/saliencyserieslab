
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
# from dtaidistance import dtw
import torch.nn as nn
import numpy as np
import torch


def distance(original, perturbed, method='dtw', sigma=1.0):
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
    similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    return similarity


def perturb_data(timeseries, num_samples=1000, segment_size=10, perturbation_ratio=0.3, adjacency_prob=0.7):
    """
    Create perturbed samples of a time series with adjacent segments more likely to be perturbed together.
    
    :param timeseries: Original time series data (1D numpy array)
    :param num_samples: Number of perturbed samples to generate
    :param segment_size: Size of each segment
    :param perturbation_ratio: Ratio of segments to perturb
    :param adjacency_prob: Probability of perturbing an adjacent segment
    :return: Perturbed samples and their binary representations
    """
    
    # Ensure the timeseries length is divisible by segment_size
    padded_length = ((len(timeseries) - 1) // segment_size + 1) * segment_size
    padded_timeseries = np.pad(timeseries, (0, padded_length - len(timeseries)), mode='constant', constant_values=0)
    
    # Convert to float to allow for noise addition
    padded_timeseries = padded_timeseries.astype(float)
    
    num_segments = padded_length // segment_size
    num_perturb = int(num_segments * perturbation_ratio)
    
    perturbed_samples = []
    binary_representations = []
    
    for _ in range(num_samples):
        # Initialize perturbation mask
        perturb_mask = np.zeros(num_segments, dtype=bool)
        
        # Start with a random segment
        current_segment = np.random.randint(0, num_segments)
        perturb_mask[current_segment] = True
        
        # Perturb adjacent segments with higher probability
        while np.sum(perturb_mask) < num_perturb:
            if np.random.random() < adjacency_prob:
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
            start = idx * segment_size
            end = start + segment_size
            
            # Randomly choose a perturbation method
            method = np.random.choice(['zero', 'noise', 'shuffle'])
            
            if method == 'zero':
                perturbed[start:end] = 0
            elif method == 'noise':
                perturbed[start:end] += np.random.normal(0, np.std(timeseries), segment_size)
            elif method == 'shuffle':
                np.random.shuffle(perturbed[start:end])
            
            binary_rep[idx] = 0
        
        perturbed_samples.append(perturbed[:len(timeseries)])
        binary_representations.append(binary_rep)
    
    return np.array(perturbed_samples), np.array(binary_representations)



def lime_explainer(model : nn.Module,
                   x : np.ndarray,
                   y : np.ndarray,
                   perturbation_ratio : float=0.3, 
                   adjacency_prob : float=0.7,
                   num_samples : int=1000, 
                   segment_size : int=10,
                   sigma : float=0.1):
    
    x_perturb, binary_rep = perturb_data(x, num_samples, segment_size, perturbation_ratio, adjacency_prob)
    
    x_perturb = torch.from_numpy(x_perturb).to(torch.float32).to(model.device)

    model.eval()
    with torch.no_grad():
        output = model(x_perturb)
        predictions = torch.argmax(output, dim=-1)

    weights = [distance(x, x_perturb[i], method='euclidean') for i in range(len(x_perturb))]

    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    logreg.fit(x_perturb, predictions, sample_weight=weights)

    # get logreg weights
    w = logreg.coef_[0]

    print(w)



if __name__ == '__main__':
    
    np.random.seed(42)  
    
    original_timeseries = np.array([1, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 1], dtype=float)

    perturbed_samples, binary_reps = perturb_data(original_timeseries, 
                                                num_samples=1, 
                                                segment_size=3, 
                                                perturbation_ratio=0.8,
                                                adjacency_prob=0.5)

    print("Original timeseries:", original_timeseries)
    print("Perturbed sample:   ", perturbed_samples[0])
    print("Binary representation:", binary_reps[0])

    similarity = distance(original_timeseries, perturbed_samples[0], method='euclidean')
    print("Similarity:         ", similarity)
