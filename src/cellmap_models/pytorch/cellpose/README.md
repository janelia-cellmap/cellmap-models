<!-- FILEPATH: /Users/rhoadesj/Repos/cellmap-models/src/cellmap_models/pytorch/cellpose/README.md -->
<h1 style="height: 56pt;">Finetuned <a href="https://www.cellpose.org"> Cellpose Models<img src="https://www.cellpose.org/static/images/cellpose_transparent.png" alt="cellpose logo" height=56pt></h1></a>

This directory contains scripts for downloading finetuned Cellpose models, particularly for use with the `cellpose` package. The models are trained on a variety of cell types from CellMap FIBSEM images, and can be used for segmentation of new data. These models are returned as [CellposeModel objects](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel).

## Models

### Trained on mouse tissues
**jrc_mus-epididymis-1_nuc_cp:**

**jrc_mus-epididymis-2_nuc_cp:**

**jrc_mus-granule-neurons-1_nuc_cp:** Cellpose was trained on nuclei from 12 2D slices from jrc_mus-granule-neurons-1.

**jrc_mus-granule-neurons-2_nuc_cp:** Cellpose was trained on nuclei from 10 2D slices from jrc_mus-granule-neurons-2.

**jrc_mus-granule-neurons-3_nuc_cp:** Cellpose was trained on nuclei from 21 2D slices from jrc_mus-granule-neurons-3.

**jrc_mus-guard-hair-follicle_nuc_cp:** Cellpose was trained on nuclei from 22 2D slices from jrc_mus-guard-hair-follicle.

**jrc_mus-heart-1_ecs_cp:**

**jrc_mus-heart-1_nuc_cp:**

**jrc_mus-hippocampus-1_nuc_cp:**

**jrc_mus-kidney_nuc_cp:** Cellpose was trained on nuclei from 28 2D slices from jrc_mus-kidney.

**jrc_mus-kidney-2_nuc_cp:** Cellpose was trained on nuclei from 38 2D slices from jrc_mus-kidney-2.

**jrc_mus-kidney-3_nuc_cp:**

**jrc_mus-liver-2_nuc_cp:** Cellpose was trained on nuclei from 29 2D slices from jrc_mus-liver-2.

**jrc_mus-liver-3_nuc_cp:**

**jrc_mus-meisnner-corpuscle-1_nuc_cp:** Cellpose was trained on nuclei from 21 2D slices from jrc_mus-meisnner-corpuscle-1.

**jrc_mus-meisnner-corpuscle-2_nuc_cp:** Cellpose was trained on nuclei from 27 2D slices from jrc_mus-meisnner-corpuscle-2.

**jrc_mus-pacinian-corpuscle_nuc_cp:** Cellpose was trained on nuclei from 31 2D slices from jrc_mus-pacinian-corpuscle.

**jrc_mus-pancreas-1_nuc_cp:** Cellpose was trained on nuclei from 34 2D slices from jrc_mus-pancreas-1.

**jrc_mus-pancreas-2_nuc_cp:** Cellpose was trained on nuclei from 30 2D slices from jrc_mus-pancreas-2.

**jrc_mus-pancreas-3_nuc_cp:** Cellpose was trained on nuclei from 4 2D slices from jrc_mus-pancreas-3.

**jrc_mus-pancreas-4_nuc_cp:** Cellpose was trained on nuclei from 17 2D slices from jrc_mus-pancreas-4.

**jrc_mus-skin-1_nuc_cp:**

**jrc_mus-thymus-1_nuc_cp:** 

### Trained on fly tissues
**jrc_fly-mb-z0419-20:** Cellpose was trained on nuclei from 32 2D slices from jrc_fly-mb-z0419-20.

## Usage

Once you have chosen a model based on the descriptions above, you can download its weights from the `cellmap-models` repository and use them as described below:

If you would like to load a model for your own use, you can do the following:

```python
from cellmap_models.cellpose import load_model
model = load_model('<model_name>')
```

Where `<model_name>` is the name of the model you would like to download.

```python

__If you would like to download and use a Cellpose model with the `cellpose` package or its GUI, do so by following the instructions below.__

First install the `cellpose` package:

```bash
conda activate cellmap
pip install "cellpose[gui]"
```

Then you can also download model weights from the `cellmap-models` repository and add them to your local Cellpose model directory. For example, you can run the following commands:

```bash
cellmap.add_cellpose <model_name>
```

where `<model_name>` is the name of the model you would like to download, based on the descriptions above. For example, to download the `jrc_mus-pancreas-4_nuc_cp` model and add it to Cellpose, you would run:

```bash
cellmap.add_cellpose jrc_mus-pancreas-4_nuc_cp
```
