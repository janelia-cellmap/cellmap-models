from pathlib import Path
from . import models_dict
from cellmap_models.utils import download_url_to_file
import torch


def load_model(
    model_name: str,
    base_path: str = f"{Path(__file__).parent}/models",
    device: str = "cuda",
):
    """Load model

    Args:
        model_name (str): model name
        base_path (str, optional): base path to store Torchscript model. Defaults to "./models".
        device (str, optional): device. Defaults to "cuda".

    Returns:
        model: model
    """
    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not available. Available models are {list(models_dict.keys())}."
        )
    if not (base_path / f"{model_name}.pth").exists():
        print(f"Downloading {model_name} from {models_dict[model_name]}")
        download_url_to_file(
            models_dict[model_name], str(base_path / f"{model_name}.pth")
        )
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available. Using CPU.")
    model = torch.jit.load(str(base_path / f"{model_name}.pth"), device)
    model.eval()
    return model
