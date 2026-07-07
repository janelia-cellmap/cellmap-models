# How to Export and Publish a DaCapo Model

This guide follows `examples/export_models/export_dacapo.py` and the `cellmap_models.model_export.dacapo_model` helper.

## Prerequisites

Install this package with DaCapo, export, and Hugging Face dependencies:

```bash
pip install -e ".[dacapo,export,huggingface]"
huggingface-cli login
```

Make sure your DaCapo config and weights stores are available in the environment where you run the export. Then set the export folder:

```bash
export CELLMAP_EXPORT_FOLDER=/path/to/exported/models
```

Each exported model folder is expected to contain `metadata.json`, `README.md`, and one or more model files such as `model.pt`, `model.pt2`, `model.ts`, or `model.onnx`.

## Export a DaCapo Run

Use `export_dacapo_model(run_name, iteration)`. The helper retrieves the DaCapo run config, loads weights for the requested iteration, generates `ModelMetadata`, writes metadata, and exports the model binaries.

```python
import os

import cellmap_models.model_export.config as c
from cellmap_models.model_export.dacapo_model import export_dacapo_model
from cellmap_models.model_export.cellmap_model import CellmapModel

c.EXPORT_FOLDER = os.environ["CELLMAP_EXPORT_FOLDER"]

run_name = "v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0"
iteration = 45000

export_dacapo_model(run_name, iteration)

folder_path = os.path.join(c.EXPORT_FOLDER, run_name)
CellmapModel(folder_path)
```

The current DaCapo helper may prompt for metadata fields that are not filled automatically, such as author, description, or spatial dimensions. Run it in an interactive shell, or adapt the helper for a fully non-interactive batch export.

You can also use the installed prompt-based CLI:

```bash
cellmap-export-dacapo
```

## Optional Format Checks

Check the TorchScript export:

```python
import os
import torch

folder_path = "/path/to/exported/models/MODEL_NAME"
model = torch.jit.load(os.path.join(folder_path, "model.ts"))
model.eval()
```

If `model.onnx` exists, check it with ONNX Runtime:

```python
import os

import numpy as np
import onnxruntime as ort

folder_path = "/path/to/exported/models/MODEL_NAME"
onnx_file = os.path.join(folder_path, "model.onnx")

session = ort.InferenceSession(onnx_file)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dummy_input = np.random.randn(1, 1, 288, 288, 288).astype(np.float32)
result = session.run([output_name], {input_name: dummy_input})
print(np.array(result).shape)
```

The exporter attempts multiple formats. If ONNX export fails in the local environment, the folder can still be valid as long as `metadata.json` and at least one of `model.pt`, `model.pt2`, `model.ts`, or `model.onnx` exists.

## Publish to Hugging Face

After validating the exported folder, push it to the `cellmap` Hugging Face organization:

```python
import os

from cellmap_models.model_export import push_to_huggingface

folder_path = "/path/to/exported/models/MODEL_NAME"
model_name = os.path.basename(folder_path)

push_to_huggingface(
    folder_path=folder_path,
    repo_id=f"cellmap/{model_name}",
    commit_message=f"Upload {model_name}",
)
```

The helper regenerates `README.md` from `metadata.json`, creates the model repo if needed, and uploads the whole folder.

## Batch Export and Publish

```python
import os

import cellmap_models.model_export.config as c
from cellmap_models.model_export import push_to_huggingface
from cellmap_models.model_export.dacapo_model import export_dacapo_model
from cellmap_models.model_export.cellmap_model import CellmapModel

c.EXPORT_FOLDER = os.environ["CELLMAP_EXPORT_FOLDER"]

models = {
    "MODEL_RUN_NAME_A": 100000,
    "MODEL_RUN_NAME_B": 45000,
}

for run_name, iteration in models.items():
    export_dacapo_model(run_name, iteration)
    folder_path = os.path.join(c.EXPORT_FOLDER, run_name)
    CellmapModel(folder_path)
    push_to_huggingface(folder_path, repo_id=f"cellmap/{run_name}")
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
