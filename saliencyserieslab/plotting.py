import datetime
import os

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


def plot_weighted_graph(x : np.ndarray, w : np.ndarray, save_path : str):

    assert len(x.shape) == 1, "x must be a 1D numpy array"
    assert len(w.shape) == 1, "w must be a 1D numpy array"

    # Create a DataFrame and sort by x

    # interpolate x and _x
    x1 = np.linspace(0, x.shape[0], num = x.shape[0])
    x2 = np.linspace(0, 1000, num = 1000)
    y1 = x
    
    f = interp1d(x1, y1) # interpolate time series
    fcas = interp1d(x1, w) # interpolate weight color
    weight = fcas(x2) # convert vector of original weight vect

    df = pd.DataFrame({'x': x2, 'y': f(x2), 'weight': weight})
    # df = df.sort_values('x')

    # Create a line collection
    points = np.array([df['x'], df['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a continuous norm to map from data points to colors
    norm = Normalize(df['weight'].min(), df['weight'].max())
    lc = LineCollection(segments, cmap='jet', norm=norm)

    # Set the values used for colormapping
    lc.set_array(df['weight'])
    lc.set_linewidth(2)

    # Add the collection to the axis
    line = ax.add_collection(lc)

    # Set the axis limits
    ax.set_xlim(df['x'].min(), df['x'].max())
    ax.set_ylim(df['y'].min()-0.1, df['y'].max()+0.1)

    # Add a colorbar
    cbar = fig.colorbar(line, ax=ax, orientation = 'vertical')
    cbar.set_label('Weight')

    # Customize the plot
    
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


def plot_simple_weighted(
        ts : np.ndarray, 
        w : np.ndarray, 
        save_path : str,
        modelname : str,
        explainername : str,
        dataset : str,
        ):
    
    max_length1, max_length2 = ts.shape[0], 10000
    
    x1 = np.linspace(0,max_length1,num = max_length1)
    x2 = np.linspace(0,max_length1,num = max_length2)
    y1 = ts

    f = interp1d(x1, y1) # interpolate time series
    fcas = interp1d(x1, w) # interpolate weight color
    weight = fcas(x2) # convert vector of original weight vector to new weight vector

    plt.scatter(x2,f(x2), c = weight, cmap = 'jet', marker='.', s= 1,vmin=0,vmax = 1)
    # plt.xlabel('Explanation for index %d, dataset %s' %(idx, ds))
    cbar = plt.colorbar(orientation = 'vertical')

    plt.title('({} - {} - {})'.format(modelname, explainername, dataset))

    plt.savefig(save_path)