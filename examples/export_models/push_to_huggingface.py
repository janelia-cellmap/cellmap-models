# %%
from cellmap_models.model_export import push_to_huggingface
import os

models_dir = os.environ["CELLMAP_MODELS_DIR"]  # e.g. export CELLMAP_MODELS_DIR=/path/to/models
org = "cellmap"

models = [
    "fly_organelles_run07_432000","fly_organelles_run07_700000","fly_organelles_run08_438000",
]

# %%
for model_name in models:
    folder_path = os.path.join(models_dir, model_name)
    repo_id = f"{org}/{model_name}"
    print(f"Pushing {folder_path} -> {repo_id}")
    push_to_huggingface(folder_path, repo_id)

# %%
