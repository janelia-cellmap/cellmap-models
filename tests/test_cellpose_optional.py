import subprocess
import sys


def test_cellpose_is_not_required_to_import_cellmap_models():
    code = """
import sys
sys.modules[\'cellpose\'] = None

import cellmap_models
import cellmap_models.pytorch
from cellmap_models.pytorch.cellpose import (
    add_model,
    download_checkpoint,
    get_model,
    load_model,
    models_dict,
    models_list,
)

assert models_list == list(models_dict)
assert all(callable(function) for function in (
    add_model, download_checkpoint, get_model, load_model
))
assert \'cellpose\' not in sys.modules or sys.modules[\'cellpose\'] is None
"""
    subprocess.run([sys.executable, "-c", code], check=True)
