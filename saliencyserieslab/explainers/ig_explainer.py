from typing import Optional, Tuple, Union
import datetime
import pickle

import numpy as np
from tqdm import tqdm

from saliencyserieslab.load_sktime_classifier import SktimeClassifier
from saliencyserieslab.plotting import plot_weighted_graph


class IntegratedGradients:
    def __init__(self, model, n_steps: int = 50):
        """
        Initialize Integrated Gradients for single instance interpretation.
        
        Args:
            model: Sktime classifier model with predict_proba function
            n_steps: Number of steps for integral approximation
        """
        self.model = model
        self.n_steps = n_steps
        
    def generate_baseline(self, input_data: np.ndarray) -> np.ndarray:
        """
        Generate baseline (zeros) with same shape as input.
        
        Args:
            input_data: Input time series of shape (time_steps,)
            
        Returns:
            Baseline array of same shape as input
        """
        return np.zeros_like(input_data)
    
    def interpolate_inputs(self, input_data: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        Generate interpolated points between baseline and input.
        
        Args:
            input_data: Input time series (time_steps,)
            baseline: Baseline data (typically zeros)
            
        Returns:
            Interpolated inputs between baseline and input
        """
        # Generate steps from 0 to 1
        alphas = np.linspace(0, 1, self.n_steps)
        
        # Compute interpolated inputs: shape (n_steps, time_steps)
        delta = input_data - baseline
        interpolated = baseline + alphas[:, None] * delta
        
        return interpolated
    
    def compute_gradients(self, interpolated_inputs: np.ndarray, target_class: int) -> np.ndarray:
        """
        Compute numerical gradients for interpolated inputs using finite differences.
        
        Args:
            interpolated_inputs: Array of interpolated inputs (n_steps, time_steps)
            target_class: Index of the target class to explain
            
        Returns:
            Gradients for each interpolated input
        """
        epsilon = 1e-6
        n_steps, time_steps = interpolated_inputs.shape
        gradients = np.zeros_like(interpolated_inputs)
        
        # Compute numerical gradients using finite differences

        with tqdm(total=int(n_steps*time_steps)) as bar:
        
            for s in tqdm(range(n_steps)):
                base_input = interpolated_inputs[s]
                base_pred = self.model.predict_proba(base_input[None, :])[0, target_class]  # Get probability for target class
                
                # Compute gradient for each time step
                for t in range(time_steps):
                    # Forward difference
                    perturbed_input = base_input.copy()
                    perturbed_input[t] += epsilon
                    forward_pred = self.model.predict_proba(perturbed_input[None, :])[0, target_class]
                    
                    # Compute gradient for target class
                    gradients[s, t] = (forward_pred - base_pred) / epsilon

                    bar.update(1)
                    
        return gradients
    
    def explain(
        self, 
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute integrated gradients for a single time series.
        
        Args:
            input_data: Input time series (time_steps,)
            target_class: Class index to explain (if None, uses predicted class)
            baseline: Optional baseline array (if None, zeros will be used)
            
        Returns:
            Tuple of (attributions, approximation error)
        """
        # Ensure input is 1D
        if input_data.ndim != 1:
            raise ValueError("Input must be a single time series (1D array)")
            
        if baseline is None:
            baseline = self.generate_baseline(input_data)
            
        # If target_class not specified, use predicted class
        if target_class is None:
            target_class = np.argmax(self.model.predict_proba(input_data[None, :])[0])
            
        # Generate interpolated inputs
        interpolated_inputs = self.interpolate_inputs(input_data, baseline)
        
        # Compute gradients
        gradients = self.compute_gradients(interpolated_inputs, target_class)
        
        # Compute integral using trapezoidal rule
        grads_sum = (gradients[:-1] + gradients[1:]) / 2.0
        avg_grads = np.mean(grads_sum, axis=0)  # (time_steps,)
        
        # Compute integrated gradients
        integrated_grads = (input_data - baseline) * avg_grads
        
        # Compute approximation error
        base_pred = self.model.predict_proba(baseline[None, :])[0, target_class]
        input_pred = self.model.predict_proba(input_data[None, :])[0, target_class]
        approximation_error = np.abs(
            np.sum(integrated_grads) - 
            (input_pred - base_pred)
        ).item()
        
        return integrated_grads, approximation_error

if __name__ == "__main__":
    
    datapath = "./data/insectsound/insectsound_test_n10.pkl"
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    
    model = SktimeClassifier()
    model.load_pretrained_model("./models/inception_1")
    
    ig = IntegratedGradients(model, n_steps=2)

    timeseries = data["x"][0].reshape(-1)
    
    w, error = ig.explain(timeseries, target_class=data["y"][0])

    w = np.interp(w, (w.min(), w.max()), (0, 1))

    plot_weighted_graph(timeseries, w, f"./plots/integrated_gradients_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")





