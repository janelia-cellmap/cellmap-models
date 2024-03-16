from pathlib import Path
from cellmap_models import download_url_to_file



def download_checkpoint(checkpoint_name: str,local_folder: Path):
    """
    download models checkpoint from s3 bucket.

    Args:
        checkpoint_name (str): Name of the checkpoint file.
        local_folder (Path): Local folder to save the checkpoint.
    return:
        checkpoint_path (Path): Path to the downloaded checkpoint.
    """
    from . import models_dict, models_list, model_names  # avoid circular import

    # Make sure the checkpoint exists
    if checkpoint_name not in models_list:
        raise ValueError(
            f"Checkpoint {checkpoint_name} not found. Available checkpoints: {models_list}"
        )
    
    checkpoint_path = Path(
        local_folder/ Path(checkpoint_name.replace(".", "_"))
    ).with_suffix(".pth")
    if not checkpoint_path.exists():
        url = models_dict[checkpoint_name]
        print(f"Downloading {checkpoint_name} from {url}")
        download_url_to_file(url, checkpoint_path)
    else:
        print(f"Checkpoint {checkpoint_name} found at {checkpoint_path}")

    return checkpoint_path
