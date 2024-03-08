from .add_model import add_model
from .load_model import load_model
from .get_model import get_model

models_dict = {
    "jrc_mus-epididymis-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-epididymis-1_nuc_cp",
    "jrc_mus-epididymis-2_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-epididymis-2_nuc_cp",
    "jrc_mus-heart-1_ecs_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-heart-1_ecs_cp",
    "jrc_mus-heart-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-heart-1_nuc_cp",
    "jrc_mus-hippocampus-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-hippocampus-1_nuc_cp",
    "jrc_mus-kidney-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-kidney-3_nuc_cp",
    "jrc_mus-liver-3_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-liver-3_nuc_cp",
    "jrc_mus-pancreas-4_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-pancreas-4_nuc_cp",
    "jrc_mus-skin-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-skin-1_nuc_cp",
    "jrc_mus-thymus-1_nuc_cp": "https://github.com/janelia-cellmap/cellmap-models/releases/download/2024.03.08/jrc_mus-thymus-1_nuc_cp",
}

models_list = list(models_dict.keys())
