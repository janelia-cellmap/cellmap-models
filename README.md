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
import cellmap_models.cosem as cosem_models
print(cosem_models.models_list)
```
This will list the available models. To load a specific model, use the `load_model` function:
```python
model = cosem_models.load_model('setup04/1820500')
```

More information on each set of models and how to use them is available in the `README.md` file in the corresponding subdirectory.
