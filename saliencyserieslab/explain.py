import torch

# from lime import lime_explainer


def load_model(model_path: str):
    
    model = torch.load(model_path)
    model.eval()
    return model