import lazy_loader as lazy

# Lazy-load submodules
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "cellpose": [
            "add_model",
            "load_model",
            "get_model",
            "download_checkpoint",
            "models_dict",
            "models_list",
        ]
    },
)

from . import cosem, untrained_models
