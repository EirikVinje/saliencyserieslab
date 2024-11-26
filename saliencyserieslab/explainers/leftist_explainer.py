from typing import Any, List
import time

import numpy as np

from saliencyserieslab.explainers.leftist.timeseries.transform_function.rand_background_transform import RandBackgroundTransform
from saliencyserieslab.explainers.leftist.timeseries.transform_function.straightline_transform import StraightlineTransform
from saliencyserieslab.explainers.leftist.timeseries.segmentator.uniform_segmentator import UniformSegmentator
from saliencyserieslab.explainers.leftist.learning_process.SHAP_learning_process import SHAPLearningProcess
from saliencyserieslab.explainers.leftist.learning_process.LIME_learning_process import LIMELearningProcess
from saliencyserieslab.explainers.leftist.timeseries.transform_function.mean_transform import MeanTransform
from saliencyserieslab.explainers.leftist.LEFTIST import LEFTIST

from saliencyserieslab.classifier import SktimeClassifier
from saliencyserieslab.load_data import UcrDataset
from saliencyserieslab.plotting import plot_weighted

class LeftistExplainer:

    def __init__(
            self,
            model : Any, 
            background : np.ndarray,
            nb_interpretable_feature : int,
            transform_name : str = "random_background",
            learning_process : str="SHAP", #LIME
            random_state : int = 42,
            ):

        self.nb_interpretable_feature = nb_interpretable_feature
        self.learning_process = learning_process
        self.transform_name = transform_name
        self.random_state = random_state
        self.background = background
        self.model = model


    def explain_instance(self, x : np.ndarray, y : np.ndarray) -> List[float]:

        if self.learning_process == 'SHAP':
            learning_process = SHAPLearningProcess(x, self.model, self.background)

        elif self.learning_process == 'LIME':
            learning_process = LIMELearningProcess(self.random_state)

        if self.transform_name == "random_background":
            self.transform = RandBackgroundTransform(x)
            self.transform.set_background_dataset(self.background)
        
        elif self.transform_name == "mean":
            self.transform = MeanTransform(x)
        
        elif self.transform_name == "straight_line":
            self.transform = StraightlineTransform(x)

        segmentator = UniformSegmentator(self.nb_interpretable_feature)
        leftist = LEFTIST(self.transform, segmentator, self.model, learning_process)

        w = leftist.explain(
            nb_neighbors=1000, 
            explained_instance=x, 
            explanation_size=self.nb_interpretable_feature, 
            idx_label=y,
            )
        
        if self.learning_process == 'SHAP':
            w = w[0]

        w = np.interp(w, (w.min(), w.max()), (0, 1))

        w = np.interp(
            np.linspace(0, w.shape[0] - 1, x.shape[0]),
            np.arange(w.shape[0]),
            w,
        )

        w = w.astype(np.float32)

        return w.tolist()
