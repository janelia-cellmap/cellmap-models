# cellmap-models

[![tests](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/tests.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/tests.yaml)
[![black](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/black.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/black.yaml)
[![mypy](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/mypy.yaml/badge.svg)](https://github.com/janelia-cellmap/cellmap-models/actions/workflows/mypy.yaml)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap-models/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap-models)

This package contains the models used for segmention by the CellMap project team at HHMI Janelia.

## Installation

```bash
git clone https://github.com/janelia-cellmap/cellmap-models
cd cellmap-models
conda env create -n cellmap python=3.10
conda activate cellmap
pip install .
```

## Usage

```python
import cellmap_models
```

Different models are available in the `cellmap-models` module. For example, to use the models produced by the `COSEM` pilot project team, and published as part of [Whole-cell organelle segmentation in volume electron microscopy](https://doi.org/10.1038/s41586-021-03977-3):

```python
import cellmap_models.cosem as cosem_models
print(cosem_models.models_list)
```
This will list the available models. To load a specific model, use the `load_model` function:
```python
model = cosem_models.load_model('setup04/1820500')
```

More information on each set of models and how to use them is available in the `README.md` file in the corresponding subdirectory.