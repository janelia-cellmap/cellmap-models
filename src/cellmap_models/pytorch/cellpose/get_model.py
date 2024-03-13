import os
from pathlib import Path

from cellmap_models import download_url_to_file

# from cellpose.utils import download_url_to_file


def get_model(
    model_name: str,
    base_path: str = f"{Path(__file__).parent}/models",
):
    """Add model to cellpose

    Args:
        model_name (str): model name
        base_path (str, optional): base path to store Torchscript model. Defaults to "./models".
    """
    from . import models_dict  # avoid circular import

    # download model to cellpose directory
    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not available. Available models are {list(models_dict.keys())}."
        )
    full_path = os.path.join(base_path, f"{model_name}.pt")
    if not Path(full_path).exists():
        print(f"Downloading {model_name} from {models_dict[model_name]}")
        download_url_to_file(models_dict[model_name], full_path)
    print(f"Downloaded model {model_name} to {base_path}.")
    return full_path
