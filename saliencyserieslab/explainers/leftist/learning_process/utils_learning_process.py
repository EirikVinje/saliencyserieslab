import numpy as np

__author__ = 'Mael Guilleme mael.guilleme[at]irisa.fr'

def predict_proba(neighbors, model_to_explain):
    """
    Classify the generated neighbors by the model to explain as in LIME.

    Parameters:
        neighbors (Neighbors): the neighbors.
        model_to_explain (..): the model to explain.

    Returns:
        neighbors (Neighbors): add the classification of the neighbors by the model to explain.
    """

    # classify neighbors by the model to explain
    neighbors.proba_labels = np.array(model_to_explain.predict_proba(neighbors.values))
    if len(neighbors.proba_labels[0]) == 1:
        neighbors.proba_labels = np.array([np.array([el[0],1-el[0]]) for el in neighbors.proba_labels])
    return neighbors

def reconstruct(neighbors, transform):
    """
    Build the values of the neighbors in the original data space of the instance to explain.
    Store the values into neighbors value as a dictionary.

    Parameters:
        neighbors_masks (np.ndarray): masks of the neighbors.
        transform (Transform): the transform function.

    Returns:
        neighbors_values (np.ndarray): values of the neighbors in the original data space of the instance to explain.
    """
    neighbors_values = np.apply_along_axis(transform.apply, 1, neighbors.masks)

    dict_neighbors_value = {}
    for idx in range(len(neighbors_values)):
        dict_neighbors_value[idx] = neighbors_values[idx]

    neighbors.values = dict_neighbors_value

    return neighbors_values


