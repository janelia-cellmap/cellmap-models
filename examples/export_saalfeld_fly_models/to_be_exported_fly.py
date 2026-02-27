#%%
import torch
from fly_organelles.model import StandardUnet
import numpy as np
from cellmap_models.model_export.generate_metadata import ModelMetadata, export_metadata, get_export_folder
from cellmap_models.model_export.export_model import export_torch_model
import cellmap_models.model_export.config as c
c.EXPORT_FOLDER = "/groups/cellmap/cellmap/zouinkhim/models/saalfeldlab/fly"
import os
os.chdir(c.EXPORT_FOLDER)

#%%
models = {"run07":{700000:"/nrs/saalfeld/heinrichl/fly_organelles/run07/model_checkpoint_700000",
                   432000:"/nrs/saalfeld/heinrichl/fly_organelles/run07/model_checkpoint_432000"},
          "run08":{438000:"/nrs/saalfeld/heinrichl/fly_organelles/run08/model_checkpoint_438000"}}


#%%
# pip install fly-organelles

input_voxel_size = np.array((8, 8, 8))
output_voxel_size = np.array((8, 8, 8))
input_shape = np.array((178, 178, 178)) 
output_shape = np.array((56, 56, 56))
inference_input_shape = input_shape
infernece_output_shape = output_shape
author = "Larissa Heinrich"
description = "Fly organelles segmentation model"

classes_names = ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]
block_shape = np.array((56, 56, 56,8))
output_channels = 8
#%%

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

# %%
# print("model loaded",model)
# %%
for run_name, iterations in models.items():
    for iteration, checkpoint_path in iterations.items():
        name = f"fly_organelles_{run_name}_{iteration}"
        print(f"Exporting model {name} from {checkpoint_path}")
        model = load_eval_model(output_channels, checkpoint_path)
        # print("model loaded",model)
        description = f"Fly organelles segmentation model {run_name} iteration {iteration}"
        


        metadata = ModelMetadata(
            model_name=name,
            iteration=iteration,
            model_type=model.__class__.__name__,
            framework="torch",
            in_channels=1,
            spatial_dims=3,
            out_channels=output_channels,
            channels_names=classes_names,
            inference_input_shape=inference_input_shape,
            inference_output_shape=infernece_output_shape,
            input_shape=input_shape,
            output_shape=output_shape,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            author=author,
            description=description,
        )
        input_shape = (1, 1, *inference_input_shape)

        export_metadata(metadata)
        export_torch_model(model, input_shape, os.path.join(get_export_folder(), name))

