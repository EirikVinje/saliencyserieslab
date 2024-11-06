from typing import Optional, Tuple, Union
import numpy as np

from saliencyserieslab.load_sktime_classifier import SktimeClassifier



class IntegratedGradients:
    def __init__(self, model, n_steps: int = 50):
        """
        Initialize Integrated Gradients for Sktime classifier.
        
        Args:
            model: Sktime classifier model with predict function
            n_steps: Number of steps for integral approximation
        """
        self.model = model
        self.n_steps = n_steps
        
    def generate_baseline(self, input_data: np.ndarray) -> np.ndarray:
        """
        Generate baseline (typically zeros) with same shape as input.
        
        Args:
            input_data: Input time series data of shape (batch_size, time_steps)
            
        Returns:
            Baseline tensor of same shape as input
        """
        return np.zeros_like(input_data)
    
    def interpolate_inputs(self, input_data: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        Generate interpolated points between baseline and input.
        
        Args:
            input_data: Input time series data (batch_size, time_steps)
            baseline: Baseline data (typically zeros)
            
        Returns:
            Interpolated inputs between baseline and input
        """
        # Generate steps from 0 to 1
        alphas = np.linspace(0, 1, self.n_steps)
        
        # Compute interpolated inputs: shape (batch_size, n_steps, time_steps)
        delta = input_data - baseline
        interpolated = baseline[:, None] + alphas[None, :, None] * delta[:, None]
        
        return interpolated
    
    def compute_gradients(self, interpolated_inputs: np.ndarray) -> np.ndarray:
        """
        Compute numerical gradients for each interpolated input using finite differences.
        
        Args:
            interpolated_inputs: Array of interpolated inputs (batch_size, n_steps, time_steps)
            
        Returns:
            Gradients for each interpolated input
        """
        epsilon = 1e-6
        batch_size, n_steps, time_steps = interpolated_inputs.shape
        gradients = np.zeros_like(interpolated_inputs)
        
        # Compute numerical gradients using central differences
        for b in range(batch_size):
            for s in range(n_steps):
                base_input = interpolated_inputs[b, s]
                base_pred = self.model.predict_proba(base_input[None, :])
                
                # Compute gradient for each time step
                for t in range(time_steps):
                    # Forward difference
                    perturbed_input = base_input.copy()
                    perturbed_input[t] += epsilon
                    forward_pred = self.model.predict_proba(perturbed_input[None, :])
                    
                    # Compute gradient
                    gradients[b, s, t] = (forward_pred - base_pred) / epsilon
                    
        return gradients
    
    def compute_integrated_gradients(
        self, 
        input_data: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute integrated gradients for input data.
        
        Args:
            input_data: Input time series data (batch_size, time_steps)
            baseline: Optional baseline array (if None, zeros will be used)
            
        Returns:
            Tuple of (integrated gradients, approximation error)
        """
        if baseline is None:
            baseline = self.generate_baseline(input_data)
            
        # Generate interpolated inputs
        interpolated_inputs = self.interpolate_inputs(input_data, baseline)
        
        # Compute gradients
        gradients = self.compute_gradients(interpolated_inputs)
        
        # Compute integral using trapezoidal rule
        grads_sum = (gradients[:, :-1] + gradients[:, 1:]) / 2.0
        avg_grads = np.mean(grads_sum, axis=1)  # (batch_size, time_steps)
        
        # Compute integrated gradients
        integrated_grads = (input_data - baseline) * avg_grads
        
        # Compute approximation error
        base_pred = self.model.predict_proba(baseline)
        input_pred = self.model.predict_proba(input_data)
        approximation_error = np.abs(
            np.sum(integrated_grads) - 
            (input_pred - base_pred)
        ).item()
        
        return integrated_grads, approximation_error


if __name__ == "__main__":


    model = SktimeClassifier()
    model.load_pretrained_model("./models/inception_1")
    
    ig = IntegratedGradients(model.model, n_steps=50)
    
    input_data = np.random.randn(1, 100)  # Example with sequence length 100
    
    attributions, error = ig.compute_integrated_gradients(input_data)
    
    print(attributions, error)

