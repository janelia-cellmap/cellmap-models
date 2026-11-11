"""A model trained with mia-train, as a cellmap-models export.

The MIA team (HHMI AI, Hugging Face ``ai-hhmi``) publish their models as a
folder of ``config.json`` (the network's constructor arguments, the
algorithm it was trained for, its input conventions, and its classes or
affinity offsets) and ``model.safetensors``. ``load_mia_model`` builds the
network from mia-train's own code and wraps it in the cellmap-models
convention; ``export_mia_model`` traces that wrapper into an export folder,
which ``CellmapModel`` loads and cellmap-flow serves.

The wrapper:

- reads (B, 1, z, y, x) in [0, 1]. mia-train's networks read (x, y, z), so
  the input is transposed going in and the output coming out;
- takes one window (the training crop, 256³ for the lmd-vitl models) and
  answers the whole window. A vision transformer attends over its whole
  input, so two windows disagree where they meet; the metadata asks the
  server to run windows overlapping by a quarter of a window a side, on the
  chunks' corners, and blend them with a ramp (``overlap``, ``block_phase``,
  ``blend``), which took the seams away in cellmap-flow;
- for semantic_seg: one channel per class, the summed softmax probability
  of the class's leaf channels (a composite class's own channel was never a
  training target), x 255 as uint8;
- for affinity_seg: each affinity as mia-train's predictor stores it,
  ``sigmoid(0.2 * logit)``, in float32, the channels in z, y, x order (short
  range, then long), each one's offset in the metadata.

Only mia-train's network code is imported (``models.dinov3_vit3d`` and
``layers.common.dense_heads``); its algorithms and registries need its
training dependencies. Its top-level packages have generic names
(``models``, ``layers``, ``utils``, ``distributed``), so they are imported
only when a model is built, and not over another package of the same name
already loaded (Dask's ``distributed``, say).
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path

import torch

from .generate_metadata import ModelMetadata, write_metadata

MIA_FILES = ("config.json", "model.safetensors")
# Read too when the source has them: the model card (its name and licence)
# and the licence itself.
_CARD_FILES = ("README.md", "LICENSE.md")

# What this module builds and serves, by the names config.json gives them.
_NETWORKS = ("dinov3_vit3d",)
_HEADS = ("UNETRHead",)
_ALGORITHMS = ("semantic_seg", "affinity_seg")

MIA_TRAIN_REPO = "https://github.com/AI-HHMI/mia-train"
# mia-train's top-level packages that its network code imports.
_MIA_TRAIN_PACKAGES = ("models", "layers", "utils", "distributed")

# The classes cellmap-flow served of ai-hhmi/lmd-vitl-p8-cellmap-lvd: the
# organelles with a test IoU worth looking at (its model card's table).
DEFAULT_CLASSES = (
    "mito",
    "er",
    "nuc",
    "golgi",
    "ves",
    "endo",
    "lyso",
    "ld",
    "perox",
    "mt",
    "pm",
    "ecs",
)

# mia-train's AffinitySegmentation.squash: the affinities its predictor
# stores, blends and segments are sigmoid(0.2 * logit) (BANIS'
# scale_sigmoid), and its benchmark ran the mutex watershed on those.
AFFINITY_LOGIT_SCALE = 0.2

# input_voxel_size of a model trained at each dataset's own voxel size
# (native_voxel_size): only a default.
DEFAULT_VOXEL_SIZE = (8, 8, 8)


def is_mia_config(config: dict) -> bool:
    """Whether ``config`` (a parsed config.json) is a mia-train model this
    module builds and serves: a ``model`` naming one of its networks, a
    ``head`` and an ``algorithm`` it serves (semantic_seg, affinity_seg).
    False for anything else, such as the MIA team's simmim pretraining
    repos, which have no head."""
    if not isinstance(config, dict):
        return False

    def name(key):
        entry = config.get(key)
        return entry.get("name") if isinstance(entry, dict) else None

    return (
        name("model") in _NETWORKS
        and name("head") in _HEADS
        and name("algorithm") in _ALGORITHMS
    )


def _load_commit(config) -> str:
    """The mia-train commit config.json says to load the model with, or ''."""
    commit = (config.get("source") or {}).get("load_with_mia_train_commit") or ""
    return str(commit).split()[0] if str(commit).strip() else ""


def _shadowed_package():
    """(name, where) of the first of mia-train's top-level packages that is
    already loaded from outside a mia-train tree; None when none is.

    A mia-train tree is a folder holding ``models/dinov3_vit3d.py``. Imported
    over another package of the same name, mia-train's own imports would fail,
    or pick up the other package's modules."""
    for name in _MIA_TRAIN_PACKAGES:
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", None) or [])
        if not locations:  # a plain module: never mia-train's, whose are packages
            return name, getattr(module, "__file__", None) or repr(module)
        root = os.path.dirname(os.path.abspath(locations[0]))
        if not os.path.isfile(os.path.join(root, "models", "dinov3_vit3d.py")):
            return name, locations[0]
    return None


def _mia_train_network(config):
    """mia-train's (DinoVisionTransformer3D, UNETRHead), imported now.

    Raises ImportError saying how to get mia-train, at the commit
    ``config`` names, when it cannot be imported."""
    commit = _load_commit(config)
    shadowed = _shadowed_package()
    if shadowed is not None:
        name, where = shadowed
        raise ImportError(
            f"mia-train's top-level package {name!r} is already loaded in this process from "
            f"another package of that name ({where}), so mia-train's network code cannot be "
            "imported. Build the model in a process that has not imported it, e.g. "
            'python -c "from cellmap_models.model_export import export_mia_model; '
            "export_mia_model('<source>', '<folder>')\"."
        )
    try:
        from layers.common.dense_heads import UNETRHead
        from models.dinov3_vit3d import DinoVisionTransformer3D
    except ImportError as e:
        at = f"@{commit}" if commit else ""
        checkout = f" && git -C mia-train checkout {commit}" if commit else ""
        raise ImportError(
            "Building a mia-train model needs mia-train's network code: "
            f'pip install "mia-train @ git+{MIA_TRAIN_REPO}{at}", or put the src/ folder of a '
            f"clone on PYTHONPATH (git clone {MIA_TRAIN_REPO}{checkout}), as the MIA team's "
            f"model cards do. ({e})"
        ) from e
    return DinoVisionTransformer3D, UNETRHead


class MiaNetwork(torch.nn.Module):
    """mia-train's encoder and dense head, as the MIA team's model cards
    build them: (B, 1, x, y, z) in [0, 1] -> (B, head channels, x, y, z)
    logits. Its weights are ``encoder.*`` and ``decoder_out.*``, as in
    their model.safetensors."""

    def __init__(self, config):
        super().__init__()
        DinoVisionTransformer3D, UNETRHead = _mia_train_network(config)
        self.encoder = DinoVisionTransformer3D(**config["model"]["kwargs"])
        self.decoder_out = UNETRHead(**config["head"]["kwargs"])
        self.skip_layers = list(config["skip_layers"])

    def forward(self, x):
        maps = self.encoder.get_intermediate_layers(x, n=self.skip_layers, reshape=True)
        return self.decoder_out(list(maps), x)


class _Window(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def logits(self, x):
        """The network's float32 logits for ``x``, (B, 1, z, y, x), in the
        network's own (x, y, z) order. Under bf16 autocast on CUDA, as the
        MIA team ran it; float32 elsewhere."""
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self.net(x.permute(0, 1, 4, 3, 2))
        return out.float()


class SemanticWindow(_Window):
    """(B, 1, z, y, x) in [0, 1], one window -> (B, classes, z, y, x) uint8,
    the whole window: row i of ``groups`` (classes x network channels, 0/1)
    sums class i's leaf channels' softmax probabilities, x 255."""

    def __init__(self, net, groups):
        super().__init__(net)
        self.register_buffer("groups", groups)

    def forward(self, x):
        p = torch.softmax(self.logits(x), dim=1)
        q = torch.einsum("gc,bcxyz->bgzyx", self.groups, p)
        return (q * 255).round().clamp(0, 255).to(torch.uint8).contiguous()


class AffinityWindow(_Window):
    """(B, 1, z, y, x) in [0, 1], one window -> (B, offsets, z, y, x)
    float32, the whole window: the network's affinity channels ``order``
    (indices into its output), each ``sigmoid(0.2 * logit)``."""

    def __init__(self, net, order):
        super().__init__(net)
        self.register_buffer("order", torch.as_tensor(list(order), dtype=torch.long))

    def forward(self, x):
        logits = self.logits(x).index_select(1, self.order)
        affinities = torch.sigmoid(AFFINITY_LOGIT_SCALE * logits)
        return affinities.permute(0, 1, 4, 3, 2).contiguous()


def default_classes(config) -> list:
    """The classes a semantic_seg model serves when none are asked for:
    DEFAULT_CLASSES when the model has all of them, else every class its
    config marks ``tested``."""
    names = [entry["name"] for entry in config["classes"]]
    if all(name in names for name in DEFAULT_CLASSES):
        return list(DEFAULT_CLASSES)
    tested = [entry["name"] for entry in config["classes"] if entry.get("tested")]
    if not tested:
        raise ValueError(
            f"its config marks no class tested: pass classes, some of {names}"
        )
    return tested


def class_groups(config, names) -> torch.Tensor:
    """(len(names), network channels) 0/1: row i picks the leaf channels of
    class ``names[i]``, its ``members``, else its own ``channel``. A
    composite's own channel was never trained, so it is never read."""
    by_name = {entry["name"]: entry for entry in config["classes"]}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"no class {unknown} in this model; it has {list(by_name)}")
    groups = torch.zeros(len(names), int(config["head"]["kwargs"]["out_channels"]))
    for i, name in enumerate(names):
        entry = by_name[name]
        groups[i, list(entry.get("members") or [entry["channel"]])] = 1.0
    return groups


def affinity_channels(config):
    """(offsets, order): the affinity channels in z, y, x order, short range
    first ([1, 0, 0], [0, 1, 0], [0, 0, 1], then the long range), each
    offset [dz, dy, dx]; ``order`` the network channel each one is.

    config.json gives ``offsets`` in x, y, z, as the network reads its axes.
    In z, y, x order the first three channels are the nearest neighbours
    along z, y and x, as cellmap-flow takes an affinity model's to be when it
    knows them by their channel names alone."""
    offsets = config.get("offsets")
    if not offsets:
        raise ValueError("an affinity_seg config.json needs its offsets")
    head_channels = int(config["head"]["kwargs"]["out_channels"])
    if len(offsets) > head_channels:
        raise ValueError(
            f"{len(offsets)} offsets for a head of {head_channels} channels"
        )
    zyx = [[int(v) for v in reversed(offset)] for offset in offsets]
    order = sorted(
        range(len(zyx)),
        key=lambda i: (sum(abs(v) for v in zyx[i]), [-v for v in zyx[i]]),
    )
    return [zyx[i] for i in order], order


def _whole(value):
    if not float(value).is_integer():
        raise ValueError(f"a voxel size of {value} nm: the metadata takes whole nm")
    return int(value)


def _voxel_size(config):
    """(input voxel size, z, y, x in nm; native_voxel_size)."""
    size = (config.get("input") or {}).get("voxel_size_nm")
    if isinstance(size, (list, tuple)) and len(size) == 3:
        return [_whole(v) for v in reversed(size)], False
    if isinstance(size, str) and size.strip().lower().startswith("native"):
        return list(DEFAULT_VOXEL_SIZE), True
    raise ValueError(
        f"input.voxel_size_nm {size!r}: expected three numbers (x, y, z) or 'native: ...'"
    )


def _window(config):
    """The window the model runs at, z, y, x: the training crop, else the
    network's img_size."""
    crop = (config.get("input") or {}).get("training_crop") or config["model"][
        "kwargs"
    ].get("img_size")
    if crop is None:
        raise ValueError("config.json gives neither input.training_crop nor img_size")
    if isinstance(crop, int):
        crop = [crop] * 3
    return [int(v) for v in reversed(crop)]


def _hf_cache_repo(folder):
    """(repo id, revision) of a Hugging Face cache snapshot folder
    (.../models--org--name/snapshots/<commit>); (None, None) for any other."""
    parts = Path(folder).parts
    if (
        len(parts) >= 3
        and parts[-2] == "snapshots"
        and parts[-3].startswith("models--")
    ):
        return parts[-3][len("models--") :].replace("--", "/", 1), parts[-1]
    return None, None


def _resolve(source, revision):
    """(local folder holding MIA_FILES, Hugging Face repo id or None, revision or None)."""
    if os.path.isdir(source):
        if revision is not None:
            raise ValueError(
                f"revision is for a Hugging Face repo id; {source} is a local folder"
            )
        folder = os.path.abspath(source)
        repo, revision = _hf_cache_repo(folder)
    elif os.path.isabs(source) or source.startswith(".") or source.count("/") != 1:
        raise FileNotFoundError(
            f"{source} is neither a folder nor a Hugging Face repo id (org/name)"
        )
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "Loading a model from Hugging Face needs huggingface_hub: "
                'pip install "cellmap-models[mia]"'
            ) from e
        folder = snapshot_download(
            repo_id=source,
            revision=revision,
            allow_patterns=[*MIA_FILES, *_CARD_FILES],
        )
        repo = source
        revision = _hf_cache_repo(folder)[1] or revision
    missing = [f for f in MIA_FILES if not os.path.isfile(os.path.join(folder, f))]
    if missing:
        raise FileNotFoundError(
            f"{source} has no {', '.join(missing)}: a mia-train model is {', '.join(MIA_FILES)}"
        )
    return folder, repo, revision


def _model_card(folder):
    """(the front matter's top-level ``key: value`` pairs, the first ``# ``
    title) of the folder's README.md; ({}, None) without one."""
    path = os.path.join(folder, "README.md")
    if not os.path.isfile(path):
        return {}, None
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    front = {}
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            match = re.match(r"^([A-Za-z_]+):\s*(\S.*)$", line)
            if match:
                front[match.group(1)] = match.group(2).strip().strip("\"'")
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), None)
    return front, title


def _provenance(folder, repo, revision, config):
    """(model name, author, the sentence saying where it comes from and its licence)."""
    front, title = _model_card(folder)
    org = repo.split("/")[0] if repo else None
    if org == "ai-hhmi" or (
        not repo and front.get("base_model", "").startswith("ai-hhmi/")
    ):
        by = " by the MIA team, HHMI AI (Hugging Face: ai-hhmi)"
    else:
        by = f" by {org} (Hugging Face)" if repo else ""
    author = (
        f"Trained with mia-train{by}; converted to a cellmap-models export by "
        "cellmap-models (model_export.mia_model)"
    )
    if repo:
        name = repo.split("/")[-1]
        where = f"https://huggingface.co/{repo}"
        if revision:
            where += f" (revision {revision})"
    else:
        if title and re.fullmatch(r"[\w.\-]+", title):
            name = title
        else:
            name = os.path.basename(folder.rstrip(os.sep))
        where = folder

    licence = front.get("license_name") or front.get("license")
    if front.get("license_link"):
        link = front["license_link"]
        licence = f"{licence} ({link})" if licence else link
    if os.path.isfile(os.path.join(folder, "LICENSE.md")):
        file = (
            f"https://huggingface.co/{repo}/blob/{revision or 'main'}/LICENSE.md"
            if repo
            else os.path.join(folder, "LICENSE.md")
        )
        licence = f"{licence}, {file}" if licence else file

    source = config.get("source") or {}
    sentence = f"From {where}"
    if source.get("run"):
        sentence += f", mia-train run {source['run']}"
    if source.get("step") is not None:
        sentence += f" at step {source['step']}"
    if _load_commit(config):
        sentence += f"; it loads with mia-train {_load_commit(config)}"
    licence = licence or "the source model's own"
    sentence += f". Licence: {licence}, which governs this export."
    return name, author, sentence


def _intensity(config):
    """(normalization, sentence) from input.intensity_range."""
    inputs = config.get("input") or {}
    lo, hi = (float(v) for v in inputs.get("intensity_range") or (0.0, 1.0))
    scaling = inputs.get("intensity_scaling")
    sentence = f"Input (z, y, x) in [{lo:g}, {hi:g}]" + (
        f" ({scaling})" if scaling else ""
    )
    return {"method": "min_max", "min": lo, "max": hi}, sentence


def load_mia_model(
    source: str,
    *,
    revision: str | None = None,
    classes: list[str] | None = None,
) -> tuple[torch.nn.Module, ModelMetadata]:
    """The wrapper (eval mode, on the CPU) and metadata of a mia-train model.

    ``source``: a Hugging Face repo id (``ai-hhmi/lmd-vitl-p8-cellmap-lvd``)
    or a local folder holding MIA_FILES. ``revision``: a branch, tag or
    commit of the repo. ``classes``: for a semantic_seg model, the class
    names to serve, one channel each (``default_classes`` when None).

    The wrapper is a ``SemanticWindow`` or an ``AffinityWindow`` (see the
    module's docstring for what it reads and answers); the network's weights
    are loaded strictly. Raises ImportError when mia-train's network code
    cannot be imported, saying how to install it.
    """
    folder, repo, revision = _resolve(source, revision)
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if not is_mia_config(config):
        found = {
            key: entry.get("name") if isinstance(entry, dict) else entry
            for key in ("model", "head", "algorithm")
            for entry in [config.get(key) if isinstance(config, dict) else None]
        }
        raise ValueError(
            f"{source} is not a mia-train model this module serves: it needs a model named one "
            f"of {_NETWORKS}, a head named one of {_HEADS} and an algorithm named one of "
            f"{_ALGORITHMS}; its config.json has {found}"
        )
    kind = config["algorithm"]["name"]
    window = _window(config)
    voxel_size, native = _voxel_size(config)
    normalization, input_sentence = _intensity(config)
    name, author, provenance = _provenance(folder, repo, revision, config)

    # Checked before the network is built: a ViT-L takes a while.
    if kind == "semantic_seg":
        if isinstance(classes, str):
            classes = [classes]
        names = list(classes) if classes is not None else default_classes(config)
        groups = class_groups(config, names)
        offsets = None
        output_dtype = "uint8"
        output_sentence = (
            "Each channel is a class's summed softmax probability over its leaf channels (a "
            "composite class's own channel was never trained), x 255 as uint8: "
            + ", ".join(names)
            + "."
        )
    else:
        if classes is not None:
            raise ValueError(
                f"classes picks the classes of a semantic_seg model; {source} is {kind}"
            )
        offsets, order = affinity_channels(config)
        # cellmap-flow takes a channel named *_aff* to be an affinity.
        names = ["object_aff_" + "_".join(str(v) for v in o) for o in offsets]
        output_dtype = "float32"
        output_sentence = (
            "Each channel is an affinity as mia-train's predictor stores it, "
            f"sigmoid({AFFINITY_LOGIT_SCALE:g} x logit), in float32: the probability that a "
            "voxel and the voxel at its channel's offset ([dz, dy, dx], affinity_offsets) belong "
            f"to the same object; offsets {offsets}."
        )

    net = MiaNetwork(config)
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            'Reading model.safetensors needs safetensors: pip install "cellmap-models[mia]"'
        ) from e
    try:
        net.load_state_dict(load_file(os.path.join(folder, "model.safetensors")))
    except RuntimeError as e:
        built_by = getattr(sys.modules.get("models"), "__path__", ["?"])[0]
        raise RuntimeError(
            f"{source}'s weights do not fit the network built by the mia-train at "
            f"{built_by}; its config.json says to load it with mia-train "
            f"{_load_commit(config) or '(no commit given)'}: {e}"
        ) from e

    if kind == "semantic_seg":
        wrapper = SemanticWindow(net, groups)
    else:
        wrapper = AffinityWindow(net, order)

    quarter = [w // 4 for w in window]
    overlap = quarter[0] if len(set(quarter)) == 1 else quarter
    shape = "x".join(str(w) for w in window)
    if native:
        voxel_sentence = (
            "Trained on each dataset at its own voxel size: run it at the data's native "
            f"resolution ({'x'.join(str(v) for v in voxel_size)} nm is only a default)."
        )
    else:
        voxel_sentence = (
            f"Trained at {'x'.join(str(v) for v in voxel_size)} nm (z, y, x)."
        )
    kwargs = config["model"]["kwargs"]
    description = " ".join(
        [
            f"{name}, a mia-train {kind} model, converted to a cellmap-models export.",
            provenance,
            f"{input_sentence}, transposed to the network's (x, y, z).",
            f"One {shape} window in, the whole window out: the network attends over all of its "
            "input, so two windows disagree where they meet, and the server should run windows "
            f"overlapping by {overlap} voxels a side, on the chunks' corners, blended with a ramp.",
            output_sentence,
            voxel_sentence,
        ]
    )
    metadata = ModelMetadata(
        model_name=name,
        model_type=(
            f"mia-train {config['model']['name']} (embed_dim {kwargs.get('embed_dim')}, "
            f"depth {kwargs.get('depth')}, patch {kwargs.get('patch_size')}) + "
            f"{config['head']['name']}"
        ),
        framework="torch",
        spatial_dims=3,
        in_channels=int(kwargs.get("in_chans", 1)),
        out_channels=len(names),
        iteration=(config.get("source") or {}).get("step"),
        input_voxel_size=voxel_size,
        output_voxel_size=voxel_size,
        channels_names=names,
        input_shape=window,
        output_shape=window,
        inference_input_shape=window,
        inference_output_shape=window,
        input_normalization=normalization,
        inference_output_activation={"name": "identity", "apply_after_model": False},
        author=author,
        description=description,
        overlap=overlap,
        block_phase="corner",
        blend="ramp",
        output_dtype=output_dtype,
        affinity_offsets=offsets,
        native_voxel_size=native,
    )
    return wrapper.eval(), metadata


def export_mia_model(
    source: str,
    folder: str,
    *,
    revision: str | None = None,
    classes: list[str] | None = None,
) -> str:
    """Export a mia-train model (``load_mia_model``'s ``source``,
    ``revision`` and ``classes``) to ``folder``, and return ``folder``.

    Writes ``model.ts`` (TorchScript, traced at the model's window),
    ``model.pt2`` (torch.export) when torch.export can export the model,
    skipped with a printed note when it cannot, ``metadata.json`` and
    ``README.md``. Traces on CUDA under bf16 autocast, as the MIA team run
    their models, when CUDA is available; else in float32 on the CPU, which
    warns: a ViT-L takes minutes a window there.
    """
    wrapper, metadata = load_mia_model(source, revision=revision, classes=classes)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        precision = "Traced on CUDA under bf16 autocast; the trace keeps that device: run it on CUDA."
    else:
        device = torch.device("cpu")
        warnings.warn(
            "No CUDA: tracing the mia-train model on the CPU in float32. A ViT-L takes minutes "
            "a window on a CPU, and the export will run in float32 wherever it runs."
        )
        precision = "Traced on the CPU in float32."
    wrapper = wrapper.to(device).eval()
    x = torch.rand(
        1, metadata.in_channels, *metadata.inference_input_shape, device=device
    )
    os.makedirs(folder, exist_ok=True)

    ts_file = os.path.join(folder, "model.ts")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, x, check_trace=False)
    traced.save(ts_file)
    print(f"Model saved to {ts_file}")

    ep_file = os.path.join(folder, "model.pt2")
    try:
        torch.export.save(torch.export.export(wrapper, (x,)), ep_file)
        print(f"Model saved to {ep_file}")
    except Exception as e:
        if os.path.exists(ep_file):  # an earlier export's: not this model's
            os.remove(ep_file)
        print(
            f"model.pt2 skipped: torch.export could not export the model ({type(e).__name__}: {e})"
        )

    metadata.description = f"{metadata.description} {precision}"
    write_metadata(metadata, folder)
    return folder
