import datetime
import os

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_weighted_graph(x : np.ndarray, w : np.ndarray, exp_method : str):

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
    ax.set_ylim(df['y'].min(), df['y'].max())

    # Add a colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Weight')

    # Customize the plot
    ax.set_title('Weighted Lineplot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.savefig(f"./plots/{exp_method}_explain_{timestamp}.png")

