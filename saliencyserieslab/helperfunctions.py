from scipy import signal
import numpy as np




def reduce_dims(ts, n, method='mean', normalize=False):
        """
        Reduce the dimensionality of a timeseries while preserving important information.

        Parameters:
        ts (np.ndarray): Input timeseries array
        n (int): Desired output length
        method (str): Reduction method ('mean', 'max', 'min', 'median', 'decimate', 'peaks')
        normalize (bool): Whether to normalize the reduced timeseries

        Returns:
        np.ndarray: Reduced timeseries
        """
        if len(ts) < n:
            raise ValueError("Output length must be smaller than input length")

        if method == 'decimate':
            # Decimation with filtering to prevent aliasing
            factor = len(ts) // n
            if factor > 1:
                reduced_ts = signal.decimate(ts, factor, n=int(np.ceil(len(ts)/n)))
            else:
                reduced_ts = ts
        elif method == 'peaks':
            # Preserve local maxima and interpolate
            peaks, _ = signal.find_peaks(ts)
            if len(peaks) > n:
                # If we have more peaks than desired points, select most prominent ones
                peak_values = ts[peaks]
                sorted_indices = np.argsort(peak_values)[-n:]
                selected_peaks = peaks[sorted_indices]
            else:
                selected_peaks = peaks

            # Create new x-axis points evenly spaced
            x_original = np.linspace(0, len(ts)-1, len(ts))
            x_new = np.linspace(0, len(ts)-1, n)
            reduced_ts = np.interp(x_new, x_original[selected_peaks], ts[selected_peaks])
        else:
            # Split into n segments and apply reduction function
            splits = np.array_split(ts, n)

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

        # Normalize the reduced timeseries
        if normalize:
            reduced_ts = (reduced_ts - reduced_ts.mean()) / reduced_ts.std()

        reduced_ts = reduced_ts.round(decimals=3)

        return reduced_ts