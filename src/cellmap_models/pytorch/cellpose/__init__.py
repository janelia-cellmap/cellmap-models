import lazy_loader as lazy


# Keep Cellpose itself out of the import path until one of these helpers is
# actually used. Cellpose is an optional dependency of cellmap-models.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "add_model": ["add_model"],
        "load_model": ["load_model"],
        "get_model": ["get_model"],
        "download_checkpoint": ["download_checkpoint"],
    },
)

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
