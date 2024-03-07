import numpy as np
from pathlib import Path
from cellmap.models import download_url_to_file

# voxel size parameters
voxel_size_output = np.array((4,) * 3)
voxel_size_input = np.array((8,) * 3)

# network parameters
padding = "valid"
constant_upsample = True
feature_widths_down = [12, 12 * 6, 12 * 6**2, 12 * 6**3]
feature_widths_up = [12 * 6, 12 * 6, 12 * 6**2, 12 * 6**3]
downsampling_factors = [(2,) * 3, (3,) * 3, (3,) * 3]
kernel_sizes_down = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
]
kernel_sizes_up = [[(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3]]

# additional network parameters for upsampling network
upsample_factor = tuple(voxel_size_input / voxel_size_output)
final_kernel_size = [(3,) * 3, (3,) * 3]
final_feature_width = 12 * 6

classes_out = 2

# download pretrained model checkpoints from s3
urls = {
    "1440000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup28/1440000",
    "775000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup28/775000",
}
for name, url in urls.items():
    download_url_to_file(url, str(Path(__file__).parent / f"{name}"))
