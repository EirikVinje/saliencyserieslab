import datetime
import os

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


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


def plot_weighted(
        ts : np.ndarray, 
        w : np.ndarray, 
        modelname : str,
        explainername : str,
        dataset : str,
        save_path : str = None,
        show : bool = False,
        colormap : str = 'jet',
        ):
    
    max_length1, max_length2 = ts.shape[0], 10000
    
    x1 = np.linspace(0,max_length1,num = max_length1)
    x2 = np.linspace(0,max_length1,num = max_length2)
    y1 = ts

    f = interp1d(x1, y1) # interpolate time series
    fcas = interp1d(x1, w) # interpolate weight color
    weight = fcas(x2) # convert vector of original weight vector to new weight vector

    plt.scatter(x2,f(x2), c = weight, cmap = colormap, marker='.', s= 1,vmin=0,vmax = 1)
    # plt.xlabel('Explanation for index %d, dataset %s' %(idx, ds))
    cbar = plt.colorbar(orientation = 'vertical')

    if not show:
        plt.title('({} - {} - {})'.format(modelname, explainername, dataset))
    else:
        plt.title('{} - {}'.format(dataset, explainername))
    
    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
