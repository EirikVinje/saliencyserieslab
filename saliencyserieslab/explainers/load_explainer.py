
import numpy as np

from saliencyserieslab.explainers.shap_explainer import KernelShapExplainer, ShapExplainer
from saliencyserieslab.explainers.lime_explainer import LimeExplainer
from saliencyserieslab.explainers.leftist_explainer import LeftistExplainer





def load_explainer(
        model,
        explainer_name : str,
        test_x : np.ndarray,
        ):
    
    if explainer_name == "shapley":
        
        return ShapExplainer(
            x_background=np.zeros((1, test_x.shape[1])),
            model=model,
            )
    
    elif explainer_name == "kernelshap":
        
        return KernelShapExplainer(
            model=model,
            x_background=np.zeros((1, test_x.shape[1])),
            algorithm="linear",
            )
    
    elif explainer_name == "leftist_shap":
        
        return LeftistExplainer(
        nb_interpretable_feature=test_x.shape[1] // 4,
        learning_process="SHAP",
        background=test_x,
        random_state=42,
        model=model,
    )

    elif explainer_name == "leftist_lime":
        
        return LeftistExplainer(
        nb_interpretable_feature=test_x.shape[1] // 4,
        learning_process="LIME",
        background=test_x,
        random_state=42,
        model=model,
    )


    elif explainer_name == "lime":
        
        return LimeExplainer(
            perturbation_ratio=0.5,
            num_samples=1000,
            model=model,
            )

    else:
        raise ValueError("explainer_name must be one of [shapley, kernelshap, leftist_shap, leftist_lime, lime]")