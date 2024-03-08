import os
from pathlib import Path
from cellpose.utils import download_url_to_file


def get_model(
    model_name: str,
    base_path: str = f"{Path(__file__).parent}/models",
):
    """Add model to cellpose

    Args:
        model_name (str): model name
        base_path (str, optional): base path to store Torchscript model. Defaults to "./models".
    """
    from . import models_dict

    # download model to cellpose directory
    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not available. Available models are {list(models_dict.keys())}."
        )

    if not Path(base_path / f"{model_name}.pth").exists():
        print(f"Downloading {model_name} from {models_dict[model_name]}")
        download_url_to_file(
            models_dict[model_name], os.path.join(base_path, f"{model_name}.pth")
        )
    print("Downloaded model {model_name} to {base_path}.")
    return
