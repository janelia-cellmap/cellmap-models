#%%
# pip install fly-organelles
from funlib.geometry.coordinate import Coordinate
import numpy as np
voxel_size = (8, 8, 8)
read_shape = Coordinate((178, 178, 178)) * Coordinate(voxel_size)
write_shape = Coordinate((56, 56, 56)) * Coordinate(voxel_size)
output_voxel_size = Coordinate((8, 8, 8))
input_voxel_size = Coordinate((8, 8, 8))
#%%
import torch
from fly_organelles.model import StandardUnet
#%%
def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model

CHECKPOINT_PATH = "/nrs/saalfeld/heinrichl/fly_organelles/run08/model_checkpoint_438000"
output_channels = 8  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
model = load_eval_model(output_channels, CHECKPOINT_PATH)
block_shape = np.array((56, 56, 56,8))
# %%
# print("model loaded",model)
# %%
