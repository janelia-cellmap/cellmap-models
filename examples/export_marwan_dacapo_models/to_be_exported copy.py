#%%
import cellmap_models.model_export.config as c
c.EXPORT_FOLDER = "/groups/cellmap/cellmap/zouinkhim/models/cellmap/marwan"
import os
os.chdir(c.EXPORT_FOLDER)
#%%
run_name =  "v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1"
from dacapo.store.create_store import create_config_store

config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)
# %%
ds = run_config.datasplit_config.train_configs[0]
# %%
raw_config = ds.raw_config
gt_config = ds.gt_config
# %%
raw_config
# %%
