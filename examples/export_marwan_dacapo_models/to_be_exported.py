#%%
marwan_models = {
    # "v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1":345000,
          "v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0":45000}

from cellmap_models.model_export.dacapo_model import export_dacapo_model

# %%
import cellmap_models.model_export.config as c
c.EXPORT_FOLDER = "/groups/cellmap/cellmap/zouinkhim/models/cellmap/marwan"
import os
os.chdir(c.EXPORT_FOLDER)
#%%
for k,v in marwan_models.items():
    export_dacapo_model(k,v)
    break
# %%
# check
import onnxruntime as ort

import numpy as np

# Path to your ONNX model
k = list(marwan_models.keys())[0]
onnx_file = os.path.join(c.EXPORT_FOLDER,k,"model.onnx")
print(f"Loading model from {onnx_file}")
#%%

# Create an inference session
session = ort.InferenceSession(onnx_file)

# Get the name of the first input of the model
input_name = session.get_inputs()[0].name
print("Input name  :", input_name)
print("Input shape :", session.get_inputs()[0].shape)

# Get the name of the first output of the model
output_name = session.get_outputs()[0].name
print("Output name :", output_name)
print("Output shape:", session.get_outputs()[0].shape)

# Prepare input data as a NumPy array (make sure shapes & dtypes match your model)
dummy_input = np.random.randn(1, 1,288,288,288).astype(np.float32)

# Run inference
result = session.run([output_name], {input_name: dummy_input})  # returns a list of outputs
print("Inference result shape:", np.array(result).shape)

# %%
import torch

model_path = os.path.join(c.EXPORT_FOLDER,k,"model.ts")
# model_path = "model.ts"
model = torch.jit.load(model_path)

# 2. Switch the model to evaluation mode
model.eval()

# 3. Prepare some test input data
#    (Make sure its shape and dtype match your model's expectations)
dummy_input = np.random.randn(1, 1,288,288,288).astype(np.float32)
tensor_input = torch.from_numpy(dummy_input)
# 4. Run inference
output = model(tensor_input)

# 5. Inspect the output
print("Output shape:", output.shape)
print("Output data:", output)

# %%
