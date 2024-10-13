from typing import List
import argparse
import datetime
import logging
import json

from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch

from localdatasets import InsectDataset
from modelutils import load_state_dict
from modelutils import model_selection
from explain import plot_weighted_graph

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger('src')

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


def perturb_data(x : np.ndarray, 
                 num_samples : int=1000, 
                 segment_size : int=10, 
                 perturbation_ratio : int=0.3, 
                 adjacency_prob : int=0.7):
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
    padded_length = ((x.shape[0] - 1) // segment_size + 1) * segment_size
    padded_timeseries = np.pad(x, (0, padded_length - x.shape[0]), mode='constant', constant_values=0)
    
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
                perturbed[start:end] += np.random.normal(0, np.std(x), segment_size)
            elif method == 'shuffle':
                np.random.shuffle(perturbed[start:end])
            
            binary_rep[idx] = 0
        
        perturbed_samples.append(perturbed[:x.shape[0]])
        binary_representations.append(binary_rep)
    
    return np.array(perturbed_samples), np.array(binary_representations)


def lime_explainer(model : nn.Module,
                   config : dict,
                   data : List=[torch.tensor, torch.tensor],
                   perturbation_ratio : float=0.4, 
                   adjacency_prob : float=0.9,
                   num_samples : int=5000, 
                   segment_size : int=1,
                   sigma : float=0.1):

    x, y = data

    x = x.reshape(1, -1)

    with torch.no_grad():
        output = model(x)
        ypred = torch.argmax(output, dim=-1)
        assert ypred == y, "Model prediction is wrong"

    x = x.cpu().numpy().reshape(-1)

    x_perturb, binary_rep = perturb_data(x, num_samples, segment_size, perturbation_ratio, adjacency_prob)

    weights = [distance(x, x_perturb[i], method='euclidean') for i in range(len(x_perturb))]
    
    x_perturb = torch.from_numpy(x_perturb).to(torch.float32).to(config['device'])

    with torch.no_grad():
        output = model(x_perturb)
        predictions = torch.argmax(output, dim=-1)
        predictions = predictions.cpu().numpy()

    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(binary_rep, predictions, sample_weight=weights)

    w = logreg.coef_[0].reshape(-1)
    w = np.interp(w, (w.min(), w.max()), (0, 1))
    w = np.repeat(w, x.shape[0] // w.shape[0])
    
    logger.info("Plot saved to ./plots")
    plot_weighted_graph(x, w)


if __name__ == '__main__':
    
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/modelconfig.json', help='Path to config file')
    parser.add_argument('--model', type=str, default="./models/resnet_20241012_143542.pth", help='Path to model file, e.g ./models/resnet_20221017_092600.pth')
    args = parser.parse_args()

    config_path = args.config
    state_dict_path = args.model

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    logger.info(f'Loaded config from : {config_path}')

    evaldata = InsectDataset(config['testpath'], config['device'], config['classes'])
    eval_loader = DataLoader(evaldata, batch_size=config['batch_size'], shuffle=False)
    logger.info(f'Loaded eval data from : {config["testpath"]}')

    model = model_selection(config, n_classes=evaldata.n_classes)
    state_dict = load_state_dict(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(config['device'])
    model.eval()
    logger.info(f'Loaded model from : {state_dict_path}')

    lime_explainer(model, config, evaldata.__getitem__(0))