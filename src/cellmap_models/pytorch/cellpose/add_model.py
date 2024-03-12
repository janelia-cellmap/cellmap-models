import os
import sys
from typing import Optional
from cellpose.io import add_model as _add_model
from cellpose.models import MODEL_DIR
from .get_model import get_model


def add_model(model_name: Optional[str] = None):
    """Add model to cellpose

    Args:
        model_name (str): model name
    """
    if model_name is None:
        model_name = sys.argv[1]
    base_path = MODEL_DIR
    get_model(model_name, base_path)
    _add_model(os.path.join(base_path, f"{model_name}.pt"))
    print(
        f"Added model {model_name}. This will now be available in the cellpose model list."
    )
    return
