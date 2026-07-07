# How to Export and Publish a Fly Model

This guide follows the fly-model examples in `examples/export_models/` and exports one Hugging Face-ready model folder per checkpoint.

## Prerequisites

Install this package with the export and Hugging Face dependencies, plus the fly model package:

```bash
pip install -e ".[export,huggingface]"
pip install fly-organelles
huggingface-cli login
```

Set the export folder and checkpoint folder:

```bash
export CELLMAP_EXPORT_FOLDER=/path/to/exported/models
export CELLMAP_CHECKPOINTS_DIR=/path/to/fly/checkpoints
```

Each exported model folder is expected to contain `metadata.json`, `README.md`, and one or more model files such as `model.pt`, `model.pt2`, `model.ts`, or `model.onnx`.

## Export a Standard Fly Model

Use the standard fly-model spatial shapes in metadata, and a separate 5D
PyTorch tensor shape for export tracing:

```python
input_shape = [178, 178, 178]
output_shape = [56, 56, 56]
inference_input_shape = [378, 378, 378]
inference_output_shape = [256, 256, 256]
export_input_shape = [1, 1, *input_shape]
```

Replace `channel_names`, checkpoint paths, and metadata fields for the model you are exporting.

```python
import os

import torch
from fly_organelles.model import StandardUnet

import cellmap_models.model_export.config as c
from cellmap_models.model_export import ModelMetadata, get_export_folder
from cellmap_models.model_export.generate_metadata import export_metadata
from cellmap_models.model_export.export_model import export_torch_model
from cellmap_models.model_export.cellmap_model import CellmapModel

c.EXPORT_FOLDER = os.environ["CELLMAP_EXPORT_FOLDER"]
checkpoints_dir = os.environ["CELLMAP_CHECKPOINTS_DIR"]

model_name = "fly_organelles_run07_700000"
checkpoint_path = os.path.join(checkpoints_dir, "run07/model_checkpoint_700000")
iteration = 700000

channel_names = ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]
channels = len(channel_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_backbone = StandardUnet(channels)
checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
model_backbone.load_state_dict(checkpoint["model_state_dict"])
model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
model.to(device)
model.eval()

metadata = ModelMetadata(
    model_name=model_name,
    iteration=iteration,
    model_type="StandardUnet",
    framework="torch",
    spatial_dims=3,
    in_channels=1,
    out_channels=channels,
    channels_names=channel_names,
    input_shape=[178, 178, 178],
    output_shape=[56, 56, 56],
    inference_input_shape=[378, 378, 378],
    inference_output_shape=[256, 256, 256],
    input_voxel_size=[8, 8, 8],
    output_voxel_size=[8, 8, 8],
    author="Author Name",
    description=f"Fly organelles segmentation model, iteration {iteration}.",
    version="1.0.0",
)

folder_path = os.path.join(get_export_folder(), model_name)

# Current reliable pattern: write metadata first, then export model binaries.
# Passing metadata directly into export_torch_model can trigger an overwrite prompt
# because export_torch_model creates the target folder before export_metadata runs.
export_metadata(metadata)
export_input_shape = [1, 1, *metadata.input_shape]
export_torch_model(model, tuple(export_input_shape), folder_path)

# Validate the exported folder before publishing.
CellmapModel(folder_path)
```

The exporter attempts multiple formats. If ONNX export fails in the local environment, the folder can still be valid as long as `metadata.json` and at least one of `model.pt`, `model.pt2`, `model.ts`, or `model.onnx` exists.

## Publish to Hugging Face

Push the exported folder to the `cellmap` Hugging Face organization:

```python
from cellmap_models.model_export import push_to_huggingface

push_to_huggingface(
    folder_path="/path/to/exported/models/fly_organelles_run07_700000",
    repo_id="cellmap/fly_organelles_run07_700000",
    commit_message="Upload fly_organelles_run07_700000",
)
```

The helper regenerates `README.md` from `metadata.json`, creates the model repo if needed, and uploads the whole folder.

## Batch Publish Existing Exported Folders

```python
import os
from cellmap_models.model_export import push_to_huggingface

models_dir = os.environ["CELLMAP_EXPORT_FOLDER"]
model_names = [
    "fly_organelles_run07_432000",
    "fly_organelles_run07_700000",
    "fly_organelles_run08_438000",
]

for model_name in model_names:
    folder_path = os.path.join(models_dir, model_name)
    repo_id = f"cellmap/{model_name}"
    print(f"Pushing {folder_path} -> {repo_id}")
    push_to_huggingface(folder_path, repo_id)
```

## DOI for the Hugging Face Model

Hugging Face Hub has its own DOI service for model and dataset repositories. Use that service after the model folder has been exported, validated, and pushed to the final Hugging Face repo.

1. Push the exported model folder to Hugging Face.
2. Open the model page, for example `https://huggingface.co/cellmap/MODEL_NAME`.
3. Go to the model repository `Settings`.
4. In the `DOI` section, click `Generate DOI`.
5. Acknowledge the DOI terms. Hugging Face notes that DOI generation shares some information, including your full name, with its DOI partner DataCite.
6. Optionally edit the author list so all model contributors are credited correctly.
7. After generation, the model page shows a DOI tag/citation link in the repository header.
8. Copy the DOI into the model card if you want it visible in the README text as well.

For updated weights or a new released revision, use the same DOI section and click `Generate new DOI`. Hugging Face assigns the DOI to the current repository revision and marks the previous version DOI as outdated, so citations can point to a specific version of the model.

A model with a DOI is intended to persist. Hugging Face documents that delete, rename, and visibility-change actions are restricted after DOI generation and require contacting Hugging Face support. Generate the DOI only after the repo name, visibility, model card, license, metadata, and uploaded files are final for that release.

As of the current Hugging Face Hub docs, DOI generation is documented as a repository settings UI flow, not as a public `huggingface_hub` Python API call. Source: https://huggingface.co/docs/hub/doi
