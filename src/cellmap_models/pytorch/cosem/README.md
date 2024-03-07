# COSEM Trained PyTorch Networks

This repository contains the COSEM trained networks, converted to PyTorch. The original COSEM repository can be found [here](https://open.quiltdata.com/b/janelia-cosem-networks/tree/v0003.2/) and the original COSEM paper can be found [here](https://doi.org/10.1038/s41586-021-03977-3).

The networks have been converted to PyTorch from their original Tensorflow versions using the scripts available [here](https://github.com/pattonw/cnnectome.conversion). All models are trained on 3D data and expect input of shape `(batch_size, 1, z, y, x)`.

This repo is pip installable, simply follow the steps below in an appropriate python environment (python >= 3.7), replacing `/path/to/cosem_models` with the path to this repository on your local machine:

```bash
cd /path/to/cosem_models
pip install .
```

Then you can load a model using the following code:

```python
import cosem_models
model = cosem_models.load_model('setup04/1820500')

# The model is now ready to use
```

Each model has a separate unet backbone and single layer prediction head. The `unet` and `head` objects are both PyTorch modules and can be used as such. You can access the separate components of the model using the following code:

```python
import cosem_models
model = cosem_models.load_model('setup04/1820500')
unet = model.unet
head = model.prediction_head
```

The models' prediction heads have the following numbers of output channels:
- setup04 -  14
- setup26.1 - 3
- setup28 - 2
- setup36 - 2
- setup46 - 2

This information is also available once the model is loaded using the `model.classes_out` attribute.
Additionally, the minimum input size for each model is available using the `model.min_input_size` attribute.
The step size for increasing the input size is available using the `model.input_size_step` attribute.
And the minimum output size for each model is available using the `model.min_output_size` attribute.

The model weights we most frequently use are `setup04/1820500` and `setup04/975000`. 
