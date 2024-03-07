<!-- FILEPATH: /Users/rhoadesj/Repos/cellmap-models/src/cellmap_models/pytorch/cellpose/README.md -->
<h1 style="height: 2em;">Finetuned Cellpose Models <img src="https://www.cellpose.org/static/images/cellpose_transparent.png" alt="cellpose logo"></h1>

This directory contains finetuned scripts for downloading Cellpose models, particularly for use with the `cellpose` package. The models are trained on a variety of cell types from CellMap FIBSEM images, and can be used for segmentation of new data.

## Models

...

## Usage

Once you have chosen a model based on the descriptions above, you can download its weights from the `cellmap-models` repository and use them as described below:

If you would like to load a model for your own use, you can do the following:

```python
from cellmap_models.cellpose import load_model
model = load_model('<model_name>')
```

__If you would like to download and use a Cellpose model with the `cellpose` package or its GUI, do so by following the instructions below.__

First install the `cellpose` package:

```bash
conda activate cellmap
pip install cellpose[gui]
```

Then you can also download model weights from the `cellmap-models` repository and add them to your local `cellpose` model directory. For example, you can run the following commands:

```bash
cellmap.add_cellpose <model_name>
```

where `<model_name>` is the name of the model you would like to download, based on the descriptions above. For example, to download the `...
