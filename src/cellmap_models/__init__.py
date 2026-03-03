"""
.. include:: ../../README.md
"""

import importlib as _importlib

_LAZY_IMPORTS = {
    "download_url_to_file": ".utils",
    "cosem": ".pytorch",
    "cellpose": ".pytorch",
    "untrained_models": ".pytorch",
    "model_export": ".model_export",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        try:
            mod = _importlib.import_module(module_path, __name__)
        except ImportError:
            raise ImportError(
                f"Optional dependency required for '{name}' is not installed. "
                f"Please install the necessary extras."
            )
        return getattr(mod, name) if module_path != f".{name}" else mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
