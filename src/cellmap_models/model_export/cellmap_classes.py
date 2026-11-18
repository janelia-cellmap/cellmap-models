"""The ids of CellMap's annotation classes.

A class's id is the value its voxels have in a CellMap label volume of all
classes (``mito`` 50, ``er`` 51, ``nuc`` 37). A model whose channels are
such classes declares their ids (``ModelMetadata.class_ids``), so that a
program that labels each voxel with its likeliest class can give it the id
an annotation would.
"""

# name -> id of every class, copied from the cellmap-segmentation-challenge's
# table: the name and id of each row (name, id, members, description) of
# src/cellmap_segmentation_challenge/utils/classes.csv in
# https://github.com/janelia-cellmap/cellmap-segmentation-challenge.
CELLMAP_CLASS_IDS = {
    "ecs": 1,
    "pm": 2,
    "mito_mem": 3,
    "mito_lum": 4,
    "mito_ribo": 5,
    "golgi_mem": 6,
    "golgi_lum": 7,
    "ves_mem": 8,
    "ves_lum": 9,
    "endo_mem": 10,
    "endo_lum": 11,
    "lyso_mem": 12,
    "lyso_lum": 13,
    "ld_mem": 14,
    "ld_lum": 15,
    "er_mem": 16,
    "er_lum": 17,
    "eres_mem": 18,
    "eres_lum": 19,
    "ne_mem": 20,
    "ne_lum": 21,
    "np_out": 22,
    "np_in": 23,
    "hchrom": 24,
    "nhchrom": 25,
    "echrom": 26,
    "nechrom": 27,
    "nucpl": 28,
    "nucleo": 29,
    "mt_out": 30,
    "cent": 31,
    "cent_dapp": 32,
    "cent_sdapp": 33,
    "ribo": 34,
    "cyto": 35,
    "mt_in": 36,
    "nuc": 37,
    "vim": 38,
    "glyco": 39,
    "golgi": 40,
    "ves": 41,
    "endo": 42,
    "lyso": 43,
    "ld": 44,
    "rbc": 45,
    "eres": 46,
    "perox_mem": 47,
    "perox_lum": 48,
    "perox": 49,
    "mito": 50,
    "er": 51,
    "ne": 52,
    "np": 53,
    "chrom": 54,
    "mt": 55,
    "isg_mem": 56,
    "isg_lum": 57,
    "isg_ins": 58,
    "isg": 59,
    "cell": 60,
    "actin": 61,
    "tbar": 62,
    "bm": 63,
    "er_mem_all": 64,
    "ne_mem_all": 65,
    "cent_all": 66,
    "chloroplast_mem": 67,
    "chloroplast_lum": 68,
    "chloroplast_sg": 69,
    "chloroplast": 70,
    "vacuole_mem": 71,
    "vacuole_lum": 72,
    "vacuole": 73,
    "plasmodesmata": 74,
}


def cellmap_class_ids(names):
    """The CellMap id of each of ``names``, in order, when every one of them
    is a CellMap class (``CELLMAP_CLASS_IDS``); None when any is not, as a
    list with some ids missing would not say which channel is which, or
    when there are none."""
    names = list(names)
    if not names or any(name not in CELLMAP_CLASS_IDS for name in names):
        return None
    return [CELLMAP_CLASS_IDS[name] for name in names]
