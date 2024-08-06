from .add_model import add_model
from .load_model import load_model
from .get_model import get_model
from .download_checkpoint import download_checkpoint

models_dict = {
    "jrc_fly-mb-z0419-20_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_fly-mb-z0419-20_nuc_cp",
    "jrc_mus-epididymis-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-epididymis-1_nuc_cp",
    "jrc_mus-epididymis-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-epididymis-2_nuc_cp",
    "jrc_mus-granule-neurons-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-granule-neurons-1_nuc_cp",
    "jrc_mus-granule-neurons-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-granule-neurons-2_nuc_cp",
    "jrc_mus-granule-neurons-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-granule-neurons-3_nuc_cp",
    "jrc_mus-guard-hair-follicle_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-guard-hair-follicle_nuc_cp",
    "jrc_mus-heart-1_ecs_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-heart-1_ecs_cp",
    "jrc_mus-heart-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-heart-1_nuc_cp",
    "jrc_mus-hippocampus-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-hippocampus-1_nuc_cp",
    "jrc_mus-kidney_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-kidney_nuc_cp",
    "jrc_mus-kidney-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-kidney-2_nuc_cp",
    "jrc_mus-kidney-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-kidney-3_nuc_cp",
    "jrc_mus-liver-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-liver-2_nuc_cp",
    "jrc_mus-liver-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-liver-3_nuc_cp",
    "jrc_mus-meissner-corpuscle-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-meissner-corpuscle-1_nuc_cp",
    "jrc_mus-meissner-corpuscle-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-meissner-corpuscle-2_nuc_cp",
    "jrc_mus-pacinian-corpuscle_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06jrc_mus-pacinian-corpuscle_nuc_cp",
    "jrc_mus-pancreas-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-pancreas-1_nuc_cp",
    "jrc_mus-pancreas-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-pancreas-2_nuc_cp",
    "jrc_mus-pancreas-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.08.06/jrc_mus-pancreas-3_nuc_cp",
    "jrc_mus-pancreas-4_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-pancreas-4_nuc_cp",
    "jrc_mus-skin-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-skin-1_nuc_cp",
    "jrc_mus-thymus-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-thymus-1_nuc_cp",
}

models_list = list(models_dict.keys())
