<img src="https://raw.githubusercontent.com/janelia-cellmap/cellmap-models/main/assets/CellMapLogo2.png" alt="CellMap logo" width="85%">

## This directory contains various untrained PyTorch model architectures.

## Models

- **ResNet**: Parameterizable 2D and 3D ResNet models with a variable number of layers and channels. This model is based on the original ResNet architecture with the addition of a decoding path, which mirrors the encoder, after the bottleneck, to produce an image output.

- **UNet2D**: A simple 2D UNet model with a variable number of output channels.

- **UNet3D**: A simple 3D UNet model with a variable number of output channels.

- **ViTVNet**: A 3D VNet model with a Vision Transformer (ViT) encoder. This model is based on the original VNet architecture with the addition of a ViT encoder in place of the original convolutional encoder.

## Usage

To use these models, you can import them directly from the `cellmap_models.pytorch.untrained_models` module. For example, to import the ResNet model, you can use the following code:

```python
from cellmap_models.pytorch.untrained_models import ResNet
model = ResNet(ndim=2, input_nc=1, output_nc=3, n_blocks=18)
```
