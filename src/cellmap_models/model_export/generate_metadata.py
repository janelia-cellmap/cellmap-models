import json
from typing import List, Optional, get_origin, get_args
from pydantic import BaseModel, Field
import os
import cellmap_models.model_export.config as c

CURRENT_FORMAT_VERSION = "1"


def get_export_folder():
    if c.EXPORT_FOLDER is not None:
        return c.EXPORT_FOLDER
    print("EXPORT_FOLDER is not set in the config, checking environment variables...")
    folder = os.getenv("EXPORT_FOLDER")
    if not folder:
        folder = input(
            "Didn't find EXPORT_FOLDER, Please enter the export folder path: "
        )
        os.environ["EXPORT_FOLDER"] = folder
    return folder


class ModelMetadata(BaseModel):
    model_name: Optional[str] = Field(None, description="Name of the model")
    model_type: Optional[str] = Field(
        None, description="Type of the model, e.g., UNet or DenseNet121"
    )
    framework: Optional[str] = Field(
        None, description="Framework used, e.g., MONAI or PyTorch"
    )
    spatial_dims: Optional[int] = Field(
        None, description="Number of spatial dimensions, e.g., 2 or 3"
    )
    in_channels: Optional[int] = Field(None, description="Number of input channels")
    out_channels: Optional[int] = Field(None, description="Number of output channels")
    iteration: Optional[int] = Field(None, description="Iteration number")
    input_voxel_size: Optional[List[int]] = Field(
        None, description="Input voxel size as comma-separated values, e.g., 8,8,8"
    )
    output_voxel_size: Optional[List[int]] = Field(
        None, description="Output voxel size as comma-separated values, e.g., 8,8,8"
    )
    channels_names: Optional[List[str]] = Field(
        None,
        description="Names of the channels as comma-separated values, e.g., 'CT, PET'",
    )
    input_shape: Optional[List[int]] = Field(
        None, description="Input shape as comma-separated values, e.g., 1,1,96,96,96"
    )
    output_shape: Optional[List[int]] = Field(
        None, description="Output shape as comma-separated values, e.g., 1,2,96,96,96"
    )
    inference_input_shape: Optional[List[int]] = Field(
        None,
        description="Inference input shape as comma-separated values, e.g., 1,1,96,96,96",
    )
    inference_output_shape: Optional[List[int]] = Field(
        None,
        description="Inference output shape as comma-separated values, e.g., 1,2,96,96,96",
    )
    author: Optional[str] = Field(None, description="Author of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    version: Optional[str] = Field("1.0.0", description="Version of the model")
    format_version: Optional[str] = Field(
        CURRENT_FORMAT_VERSION, description="CellmapModel format version"
    )


def _format_list(values):
    if values is None:
        return "N/A"
    return ", ".join(str(v) for v in values)


def generate_huggingface_readme(metadata: ModelMetadata):
    """Generate a HuggingFace model card README with YAML frontmatter."""
    tags = [
        "pytorch",
        "onnx",
        "torchscript",
        "3d",
        "segmentation",
        "electron-microscopy",
        "cellmap",
    ]
    if metadata.channels_names:
        tags.extend(metadata.channels_names)

    readme = f"""---
library_name: cellmap-models
tags:
{chr(10).join(f"- {t}" for t in tags)}
license: bsd-3-clause
---

# {metadata.model_name or "Unnamed"}

{metadata.description or ""}

## Model Details

| | |
|---|---|
| **Architecture** | {metadata.model_type or "N/A"} |
| **Framework** | {metadata.framework or "N/A"} |
| **Spatial Dims** | {metadata.spatial_dims or "N/A"} |
| **Input Channels** | {metadata.in_channels or "N/A"} |
| **Output Channels** | {metadata.out_channels or "N/A"} |
| **Channel Names** | {_format_list(metadata.channels_names)} |
| **Iteration** | {metadata.iteration or "N/A"} |
| **Input Voxel Size** | {_format_list(metadata.input_voxel_size)} nm |
| **Output Voxel Size** | {_format_list(metadata.output_voxel_size)} nm |
| **Inference Input Shape** | {_format_list(metadata.inference_input_shape)} |
| **Inference Output Shape** | {_format_list(metadata.inference_output_shape)} |

## Available Formats

| File | Format | Usage |
|---|---|---|
| `model.pt` | PyTorch pickle | `torch.load("model.pt")` |
| `model.ts` | TorchScript | `torch.jit.load("model.ts")` |
| `model.onnx` | ONNX | `onnxruntime.InferenceSession("model.onnx")` |
| `metadata.json` | JSON | Model metadata |

## Usage

```bash
pip install cellmap-models
```

```python
from cellmap_models.model_export.cellmap_model import CellmapModel

model = CellmapModel("path/to/model/folder")

# Inference
output = model.ts_model(input_tensor)

# Finetuning
trainable_model = model.train()
```

Or download from this repo and load directly:

```python
from huggingface_hub import snapshot_download
from cellmap_models.model_export.cellmap_model import CellmapModel

path = snapshot_download(repo_id="{metadata.model_name or 'janelia-cellmap/model'}")
model = CellmapModel(path)
```

## Author

{metadata.author or "CellMap Project Team, HHMI Janelia"}

## Links

- [cellmap-models](https://github.com/janelia-cellmap/cellmap-models)
- [CellMap Project](https://www.janelia.org/project-team/cellmap)
"""
    return readme


def generate_readme(metadata: ModelMetadata):
    readme_content = f"""# {metadata.model_name or "Unnamed"} Model
iteration: {metadata.iteration}

## Description
{metadata.description or "N/A"}

## Model Details
- **Model Type:** {metadata.model_type or "N/A"}
- **Framework:** {metadata.framework or "N/A"}
- **Spatial Dimensions:** {metadata.spatial_dims}
- **Input Channels:** {metadata.in_channels}
- **Output Channels:** {metadata.out_channels}
- **Channel Names:** {_format_list(metadata.channels_names)}
- **Input Shape:** {_format_list(metadata.input_shape)}
- **Output Shape:** {_format_list(metadata.output_shape)}
- **Inference Input Shape:** {_format_list(metadata.inference_input_shape)}
- **Inference Output Shape:** {_format_list(metadata.inference_output_shape)}
- **Input Voxel Size:** {_format_list(metadata.input_voxel_size)}
- **Output Voxel Size:** {_format_list(metadata.output_voxel_size)}

## Author
{metadata.author or "N/A"}

## Version
{metadata.version}
"""
    return readme_content


def export_metadata(metadata: ModelMetadata, overwrite: bool = False):
    export_folder = get_export_folder()
    result_folder = os.path.join(export_folder, metadata.model_name)
    if os.path.exists(result_folder) and not overwrite:
        answer = input(f"Folder {result_folder} already exists. Overwrite? [y/N]: ")
        if answer.lower() not in ("y", "yes"):
            return
    metadata = prompt_for_missing_fields(metadata)
    os.makedirs(result_folder, exist_ok=True)
    output_file = os.path.join(result_folder, "metadata.json")
    with open(output_file, "w") as f:
        json.dump(metadata.model_dump(), f, indent=4)
    print(f"Metadata saved to {output_file}")
    readme = generate_readme(metadata)
    readme_file = os.path.join(result_folder, "README.md")
    with open(readme_file, "w") as f:
        f.write(readme)
    print(f"README saved to {readme_file}")


def _is_list_annotation(annotation) -> bool:
    """Check if a type annotation is Optional[List[...]]."""
    # Optional[X] is Union[X, None], so dig into it
    origin = get_origin(annotation)
    if origin is list or origin is List:
        return True
    args = get_args(annotation)
    return any(get_origin(a) is list or get_origin(a) is List for a in args)


def _get_list_element_type(annotation):
    """Get the element type of Optional[List[T]] -> T."""
    origin = get_origin(annotation)
    if origin is list or origin is List:
        args = get_args(annotation)
        return args[0] if args else str
    for a in get_args(annotation):
        inner_origin = get_origin(a)
        if inner_origin is list or inner_origin is List:
            inner_args = get_args(a)
            return inner_args[0] if inner_args else str
    return str


def prompt_for_missing_fields(metadata: ModelMetadata):
    for field_name, field_info in ModelMetadata.model_fields.items():
        value = getattr(metadata, field_name)
        if value is None:
            try:
                prompt_text = (
                    field_info.description or f"Enter {field_name.replace('_', ' ')}"
                )

                if _is_list_annotation(field_info.annotation):
                    user_input = input(f"{prompt_text}: ")
                    elem_type = _get_list_element_type(field_info.annotation)
                    if elem_type is str:
                        value = [item.strip() for item in user_input.split(",")]
                    else:
                        value = [
                            elem_type(item.strip()) for item in user_input.split(",")
                        ]
                elif field_info.annotation in (int, Optional[int]):
                    value = int(input(f"{prompt_text}: "))
                else:
                    value = input(f"{prompt_text}: ")
                setattr(metadata, field_name, value)
            except Exception as e:
                raise Exception(f"Error prompting for field {field_name}: {e}")

    return metadata
