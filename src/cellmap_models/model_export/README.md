# Model Export

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
    input_normalization={"method": "min_max", "min": -1.0, "max": 1.0},
    inference_output_activation={"name": "sigmoid", "apply_after_model": True},
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

Pre/post-processing metadata can be parsed programmatically:

```python
from cellmap_models.model_export import parse_activation_to_torch, parse_normalization_min_max

input_min, input_max = parse_normalization_min_max(model.metadata.input_normalization)
activation = parse_activation_to_torch(model.metadata.inference_output_activation)
```

### Load an exported model for finetuning

```python
from cellmap_models.model_export.cellmap_model import CellmapModel

cellmap_model = CellmapModel("/path/to/export/folder/my_model")
model = cellmap_model.train()
# Returns an nn.Module in train mode
# Tries torch.export (model.pt2) + unflatten first, falls back to TorchScript
```

### Export for the browser (onnxruntime-web / WebGPU)

```python
from cellmap_models.model_export import export_onnx_web

export_onnx_web(model, "export/onnx_web/my_model", "my_model", input_sizes=[148, 132], validation_input=x)
```

Writes fp32 / fp16 / int8w / int8w_fp16 ONNX files per block size, with two exact rewrites for
onnxruntime-web: ConvTranspose3d (k=s=2) → 1×1×1 conv + depth-to-space, and every Conv3d → a sum of Conv2d
over a depth-folded batch (~17× faster on WebGPU, whose Conv3d kernel is naive; `native_conv3d=True` keeps
Conv3d for CPU/CUDA runtimes, where the rewrite is slower). Each file is validated against PyTorch with
onnxruntime. See [how_to/how_to_onnx_web.md](../../../how_to/how_to_onnx_web.md).

### Initialize a small model by pruning a big one

```python
from cellmap_models.model_export import prune_funlib_unet

student.load_state_dict(prune_funlib_unet(teacher.state_dict(), student.state_dict()))
```

Structured L1 channel selection for funlib U-Nets (e.g. fly_organelles `StandardUnet`), as a fine-tuning
initialization. See [how_to/how_to_prune.md](../../../how_to/how_to_prune.md).

### Push an exported model to Hugging Face Hub

```python
from cellmap_models.model_export import push_to_huggingface

push_to_huggingface(
    folder_path="/path/to/export/folder/my_model",
    repo_id="janelia-cellmap/my-model",
)
```

Requires `pip install cellmap-models[huggingface]` or `pip install huggingface-hub`.
You must be logged in via `huggingface-cli login` first.
