import os
import json

from .generate_metadata import CURRENT_FORMAT_VERSION

# Registry mapping format_version -> module name
_VERSION_MODULES = {
    "1": "cellmap_model_v1",
}


def CellmapModel(folder_path: str):
    """
    Factory that reads format_version from metadata.json and returns
    the appropriate versioned CellmapModel class instance.

    Models without format_version are treated as version "1".
    """
    metadata_file = os.path.join(folder_path, "metadata.json")
    if not os.path.exists(metadata_file):
        raise ValueError(f"metadata.json not found in {folder_path}")

    with open(metadata_file, "r") as f:
        data = json.load(f)

    fmt = data.get("format_version", "1")

    if fmt not in _VERSION_MODULES:
        if int(fmt) > int(CURRENT_FORMAT_VERSION):
            raise ValueError(
                f"Format version {fmt} is newer than supported version {CURRENT_FORMAT_VERSION}. "
                f"Please upgrade cellmap-models: pip install --upgrade cellmap-models"
            )
        raise ValueError(f"Unknown format version: {fmt}")

    import importlib
    module = importlib.import_module(f".{_VERSION_MODULES[fmt]}", package=__package__)
    return module.CellmapModel(folder_path)


def get_huggingface_model(repo_id: str, revision: str | None = None):
    """
    Download a model from Hugging Face Hub and return a CellmapModel instance.

    Args:
        repo_id: HuggingFace repo id, e.g. "cellmap/fly_organelles_run07_432000"
        revision: Optional branch, tag, or commit hash. Defaults to latest.

    Returns:
        CellmapModel instance loaded from the downloaded snapshot.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, revision=revision)
    return CellmapModel(path)
