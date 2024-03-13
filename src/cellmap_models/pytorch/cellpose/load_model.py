import os
from pathlib import Path
from typing import Union
import torch
from .get_model import get_model
from cellpose.models import CellposeModel


def load_model(
    model_name: str,
    base_path: str = f"{Path(__file__).parent}/models",
    device: Union[str, torch.device] = "cuda",
) -> torch.nn.Module:
    """Load model

    Args:
        model_name (str): model name
        base_path (str, optional): base path to store Torchscript model. Defaults to "./models".
        device (str, optional): device. Defaults to "cuda".

    Returns:
        model: model
    """
    model_path = get_model(model_name, base_path)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available. Using CPU.")
    if isinstance(device, str):
        device = torch.device(device)

    model = CellposeModel(pretrained_model=model_path, device=device)

    print(f"{model.diam_labels} diameter labels were used for training")

    return model
