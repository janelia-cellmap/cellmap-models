# %%
import os

import torch
from fly_organelles.model import StandardUnet

from fly_organelles.run import set_weights
import torch

checkpoints_dir = os.environ["CELLMAP_CHECKPOINTS_DIR"]  # e.g. export CELLMAP_CHECKPOINTS_DIR=/path/to/checkpoints
checkpoint_p = os.path.join(checkpoints_dir, "setup_19/model_checkpoint_370000")

affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
    ]
model = StandardUnet(3)
device = torch.device("cpu")
checkpoint = torch.load(checkpoint_p, weights_only=True, map_location=device)

checkpoint_classes = [f"c_{i}" for i in range(13)]
labels = [f"c_{i}" for i in range(3)]
updated_checkpoint = set_weights(
    model, checkpoint["model_state_dict"], checkpoint_classes, labels
)
model.load_state_dict(updated_checkpoint, strict=True)

#%%
model.eval()
# %%
import torch
import cellmap_models.model_export.config as c
from cellmap_models.model_export import ModelMetadata, get_export_folder, export_torch_model
import os

c.EXPORT_FOLDER = os.environ["CELLMAP_EXPORT_FOLDER"]
#%%

metadata = ModelMetadata(
    model_name="mito_aff_unet_setup_19_worms",
    model_type="UNet",
    framework="torch",
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels_names=["mito_aff_1", "mito_aff_2", "mito_aff_3"],
    input_shape=[1, 1, 178, 178, 178],
    output_shape=[1, 3, 56, 56, 56],
    inference_input_shape=[378, 378, 378],
    inference_output_shape=[256, 256, 256],
    input_voxel_size=[16, 16, 16],
    output_voxel_size=[16, 16, 16],
    input_normalization={"method": "min_max", "min": -1.0, "max": 1.0},
    inference_output_activation={"name": "sigmoid", "apply_after_model": True},
    author="Marwan Zouinkhi",
    description="C-elegans worms affinities for mitochondria segmentation using a UNet architecture trained on setup 19 with 370k iterations.",
)

input_shape = (1, 1, 178, 178, 178)
export_torch_model(model, input_shape, os.path.join(get_export_folder(), "mito_aff_unet_setup_19_worms"), metadata=metadata)
# %%
from cellmap_models.model_export import push_to_huggingface

push_to_huggingface(
    folder_path="/groups/cellmap/cellmap/zouinkhim/exp_mito/export/mito_aff_unet_setup_19_worms",
    repo_id="cellmap/mito-aff-unet-setup-19-worms",
)
# %%
from huggingface_hub import list_models

models = list_models(author="cellmap")
for m in models:
    print(m.id)
# %%
m
# %%
from huggingface_hub import list_models, ModelCard

models = list_models(author="cellmap")
for m in models:
    try:
        card = ModelCard.load(m.id)
        print(f"=== {m.id} ===")
        print(card.text[:500])  # first 500 chars of the README
        print()
    except Exception:
        print(f"=== {m.id} === No model card")
# %%
from huggingface_hub import list_models, hf_hub_download
import json

models = list_models(author="cellmap")
for m in models:
    try:
        path = hf_hub_download(m.id, "metadata.json")
        metadata = json.load(open(path))
        print(f"{m.id}: {metadata.get('description', 'No description')}")
    except Exception as e:
        print(f"{m.id}: Could not load metadata.json ({e})")