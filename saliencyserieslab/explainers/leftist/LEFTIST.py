from saliencyserieslab.explainers.leftist.learning_process.learning_process import LearningProcess
from saliencyserieslab.explainers.leftist.learning_process.utils_learning_process import predict_proba

__author__ = 'Mael Guilleme mael.guilleme[at]irisa.fr'

class LEFTIST:
    """
    Local explainer for time series classification.

    Attributes:
        transform (python function): the function to generate neighbors representation from interpretable features.
        segmenetator (python function): the function to get the interpretable features.
        model_to_explain (python function): the model to explain, must returned proba as prediction.
        learning_process (LearningProcess): the method to learn the explanation model.
    """
    def __init__(self, transform, segmentator, model_to_explain, learning_process):
        self.neighbors = None
        self.transform = transform
        self.segmentator = segmentator
        self.model_to_explain = model_to_explain
        self.learning_process = learning_process

    def explain(self, nb_neighbors, explained_instance, idx_label=None, explanation_size=None):
        """
        Compute the explanation model.

        Parameters:
            nb_neighbors (int): number of neighbors to generate.
            explained_instance (np.ndarray): time series instance to explain
            idx_label (int): index of label to explain. If None, return an explanation for each label.
            explanation_size (int): number of feature to use for the explanations

        Returns:
            An explanation model for the desired label
        """
        # get the number of features of the simplified representation
        nb_interpretable_features, segments_interval = self.segmentator.segment(explained_instance)

        self.transform.segments_interval = segments_interval

        # generate the neighbors around the instance to explain
        self.neighbors = self.learning_process.neighbors_generator.generate(nb_interpretable_features, nb_neighbors,self.transform)

        # classify the neighbors
        self.neighbors = predict_proba(self.neighbors, self.model_to_explain)

        # build the explanation from the neighbors
        if idx_label is None:
            explanations = []
            for label in range(self.neighbors.proba_labels.shape[1]):
                explanations.append(self.learning_process.solve(self.neighbors, label, explanation_size=explanation_size))
        else:
            explanations = self.learning_process.solve(self.neighbors, idx_label, explanation_size=explanation_size)
        
        return explanations

