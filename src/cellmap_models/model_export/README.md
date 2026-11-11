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

### Export a model trained with mia-train

```python
from cellmap_models.model_export import export_mia_model

export_mia_model("ai-hhmi/lmd-vitl-p8-cellmap-lvd", "/path/to/export/lmd-vitl-p8-cellmap-lvd")
```

The source is a Hugging Face repo id or a local folder holding `config.json` and `model.safetensors`, as the
MIA team (HHMI AI, [ai-hhmi](https://huggingface.co/ai-hhmi)) publish their models: a `semantic_seg` or
`affinity_seg` model (`is_mia_config(config)` says whether a config is one). It writes `model.ts`, traced at the
model's window (256³ for the lmd-vitl models), `model.pt2` when torch.export manages, `metadata.json` and
`README.md`. With a GPU the trace is taken on CUDA under bf16 autocast, as the MIA team run their models, and runs
on CUDA; without one it is float32 on the CPU, which is slow (minutes a window for a ViT-L).

The model reads (z, y, x) in [0, 1] and answers its whole window. A vision transformer attends over all of its
input, so two windows disagree where they meet: the metadata asks the server to run windows overlapping by a quarter
of a window a side and blend them (`overlap`, `block_phase: "corner"`, `blend: "ramp"`). A `semantic_seg` model
answers one uint8 channel per class, the summed softmax probability of the class's leaf channels × 255
(`classes=[...]` picks them; twelve organelles by default). An `affinity_seg` model answers float32 affinities,
mia-train's `sigmoid(0.2 × logit)`, with each channel's [dz, dy, dx] in `affinity_offsets`. `native_voxel_size: true`
marks a model trained on each dataset at its own voxel size; `input_voxel_size` is then only a default.
`load_mia_model(source)` returns the eager wrapper and its metadata without exporting.

It needs safetensors, huggingface_hub, and mia-train's network code at the commit the model's `config.json` names
(`source.load_with_mia_train_commit`):

```bash
pip install "cellmap-models[mia]"
pip install "mia-train @ git+https://github.com/AI-HHMI/mia-train@<commit>"
```

or put the `src/` folder of a mia-train clone on `PYTHONPATH`, as the MIA team's model cards do. mia-train's
top-level packages (`models`, `layers`, `utils`, `distributed`) shadow packages of the same name, such as Dask's
`distributed`: export in a process that does not use those. The export keeps the source model's licence (the
ai-hhmi models are under the DINOv3 License); `push_to_huggingface` would label it bsd-3-clause.

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
