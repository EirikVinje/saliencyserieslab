from shap_explainer import GradientShapExplainer
from shap_explainer import KernelShapExplainer
from leftist_explainer import LeftistExplainer
from ig_explainer import IntegratedGradients
from lime_explainer import LimeExplainer


def load_explainer(
        model,
        explainer_name : str, 
        ):
    
    if explainer_name == "gradientshap":
        return GradientShapExplainer(model)
    elif explainer_name == "kernelshap":
        return KernelShapExplainer(model)
    elif explainer_name == "leftist":
        return LeftistExplainer(model)
    elif explainer_name == "ig":
        return IntegratedGradients(model)
    elif explainer_name == "lime":
        return LimeExplainer(model)
    else:
        raise ValueError("explainer_name must be one of [gradientshap, kernelshap, leftist, ig, lime]")