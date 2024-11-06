import datetime
import os

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_weighted_graph(x : np.ndarray, w : np.ndarray, save_path : str):

    assert len(x.shape) == 1, "x must be a 1D numpy array"
    assert len(w.shape) == 1, "w must be a 1D numpy array"

    _x = np.arange(0, x.shape[0])

    # Create a DataFrame and sort by x
    df = pd.DataFrame({'x': _x, 'y': x, 'weight': w})
    df = df.sort_values('x')

    # Create a line collection
    points = np.array([df['x'], df['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a continuous norm to map from data points to colors
    norm = Normalize(df['weight'].min(), df['weight'].max())
    lc = LineCollection(segments, cmap='inferno', norm=norm)

    # Set the values used for colormapping
    lc.set_array(df['weight'])
    lc.set_linewidth(2)

    # Add the collection to the axis
    line = ax.add_collection(lc)

    # Set the axis limits
    ax.set_xlim(df['x'].min(), df['x'].max())
    ax.set_ylim(df['y'].min()-0.1, df['y'].max()+0.1)

    # Add a colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Weight')

    # Customize the plot
    ax.set_title('Weighted Lineplot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.savefig(save_path)


def plot_graph(
        orig_x : np.ndarray, 
        perturbed_x : np.ndarray = None, 
        save_path : str = None,
        show : bool = False
        ):

    assert len(orig_x.shape) == 1, "orig_x must be a 1D numpy array"

    if perturbed_x is not None:
        assert len(perturbed_x.shape) == 1, "perturbed_x must be a 1D numpy array"
    
    # plot both the original and perturbed data
    fig, ax = plt.subplots()
    
    ax.plot(orig_x, label='Original')
    
    if perturbed_x is not None:
        ax.plot(perturbed_x, label='Perturbed')
    
    ax.legend()
    
    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()