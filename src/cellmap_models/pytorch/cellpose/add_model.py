from . import models_dict
from cellpose.io import _add_model
from cellpose.models import MODEL_DIR
from cellpose.utils import download_url_to_file


def add_model(model_name: str):
    """Add model to cellpose

    Args:
        model_name (str): model name
    """
    # download model to cellpose directory
    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not available. Available models are {list(models_dict.keys())}."
        )
    base_path = MODEL_DIR

    if not (base_path / f"{model_name}.pth").exists():
        print(f"Downloading {model_name} from {models_dict[model_name]}")
        download_url_to_file(
            models_dict[model_name], str(base_path / f"{model_name}.pth")
        )
    _add_model(str(base_path / f"{model_name}.pth"))
    print(
        f"Added model {model_name}. This will now be available in the cellpose model list."
    )
    return
