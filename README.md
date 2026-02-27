[![GitHub Org's stars](https://img.shields.io/github/stars/Janelia-cellmap)](https://github.com/janelia-cellmap)


<img src="https://raw.githubusercontent.com/janelia-cellmap/cellmap-models/main/assets/CellMapLogo2.png" alt="CellMap logo" width="85%">

# cellmap-models

![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/janelia-cellmap/cellmap-models/total)
![GitHub License](https://img.shields.io/github/license/janelia-cellmap/cellmap-models)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjanelia-cellmap%2Fcellmap-models%2Fmain%2Fpyproject.toml)

[![tests](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/tests.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/tests.yaml)
[![black](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/black.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/black.yaml)
[![mypy](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/mypy.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/mypy.yaml)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap-models/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap-models)

This package contains the models used for segmention by the CellMap project team at HHMI Janelia.

## Installation

We strongly recommend installing within a [conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) (or [mamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install)) environment to install the package.

```bash
conda env create -y -n cellmap python=3.10 pytorch -c pytorch
conda activate cellmap
pip install cellmap-models
```

## Usage

```python
import cellmap_models
```

Different models are available in the `cellmap-models` module. For example, to use the models produced by the `COSEM` pilot project team, and published as part of [Whole-cell organelle segmentation in volume electron microscopy](https://doi.org/10.1038/s41586-021-03977-3):

```python
import cellmap_models.pytorch.cosem as cosem_models
print(cosem_models.models_list)
```
This will list the available models. To load a specific model, use the `load_model` function:
```python
model = cosem_models.load_model('setup04/1820500')
```

More information on each set of models and how to use them is available in the `README.md` file in the corresponding subdirectory.

## Model Export

Export PyTorch models to multiple formats for inference and finetuning.

### Export Formats

Each exported model directory contains:

| File | Format | Description |
|---|---|---|
| `model.pt` | PyTorch pickle | Full model object (`torch.save`) |
| `model.pt2` | torch.export | `ExportedProgram` — supports `unflatten` for finetuning |
| `model.ts` | TorchScript | Traced model (`torch.jit.trace`) |
| `model.onnx` | ONNX | For cross-framework inference |
| `metadata.json` | JSON | Model metadata (shapes, voxel sizes, channels, etc.) |
| `README.md` | Markdown | Auto-generated model card |

### Export a DaCapo model

```python
import cellmap_models.model_export.config as c
c.EXPORT_FOLDER = "/path/to/export/folder"

from cellmap_models.model_export.dacapo_model import export_dacapo_model

export_dacapo_model("my_run_name", iteration=100000)
```

### Export any PyTorch model

```python
import torch
import cellmap_models.model_export.config as c
from cellmap_models.model_export import ModelMetadata, get_export_folder, export_torch_model
import os

c.EXPORT_FOLDER = "/path/to/export/folder"

model = ...  # any torch.nn.Module
model.eval()

metadata = ModelMetadata(
    model_name="my_model",
    model_type="UNet",
    framework="torch",
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels_names=["mito", "er"],
    input_shape=[1, 1, 96, 96, 96],
    output_shape=[1, 2, 96, 96, 96],
    inference_input_shape=[96, 96, 96],
    inference_output_shape=[96, 96, 96],
    input_voxel_size=[8, 8, 8],
    output_voxel_size=[8, 8, 8],
    author="Your Name",
    description="My segmentation model",
)

input_shape = (1, 1, 96, 96, 96)
export_torch_model(model, input_shape, os.path.join(get_export_folder(), "my_model"), metadata=metadata)
```

### Load an exported model for inference

```python
from cellmap_models.model_export.cellmap_model import CellmapModel

model = CellmapModel("/path/to/export/folder/my_model")

print(model.metadata.channels_names)

onnx_session = model.onnx_model        # ONNX Runtime session
ts_model = model.ts_model              # TorchScript model
pt_model = model.pytorch_model         # PyTorch pickle model
exported = model.exported_model         # torch.export ExportedProgram
```

### Load an exported model for finetuning

```python
from cellmap_models.model_export.cellmap_model import CellmapModel

cellmap_model = CellmapModel("/path/to/export/folder/my_model")
model = cellmap_model.train()
# Returns an nn.Module in train mode
# Tries torch.export (model.pt2) + unflatten first, falls back to TorchScript
```
