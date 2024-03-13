import os
from pathlib import Path
import torch
from .get_model import get_model


def load_model(
    model_name: str,
    base_path: str = f"{Path(__file__).parent}/models",
    device: str = "cuda",
) -> torch.nn.Module:
    """Load model

    Args:
        model_name (str): model name
        base_path (str, optional): base path to store Torchscript model. Defaults to "./models".
        device (str, optional): device. Defaults to "cuda".

    Returns:
        model: model
    """

    get_model(model_name, base_path)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available. Using CPU.")
    # TODO: load using cellpose utilities
    model = torch.jit.load(os.path.join(base_path, f"{model_name}.pt"), device)
    model.eval()
    return model
