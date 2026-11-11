import json
import os
import sys

import pytest
import torch

from cellmap_models.model_export import (
    ModelMetadata,
    export_mia_model,
    is_mia_config,
    load_mia_model,
    mia_metadata,
    write_metadata,
)
from cellmap_models.model_export.cellmap_model import CellmapModel
from cellmap_models.model_export.generate_metadata import (
    generate_huggingface_readme,
    prompt_for_missing_fields,
)
from cellmap_models.model_export.mia_model import MiaNetwork

COMMIT = "4e719562dcafc4761c3a30359899467af2ba38cb"


def tiny_config(algorithm):
    """A mia-train config.json as the MIA team publish them, for a network small
    enough to run on a CPU; a non-cubic crop, so that an axis put in the wrong
    place shows."""
    config = {
        "model": {
            "name": "dinov3_vit3d",
            "kwargs": {
                "img_size": 16,
                "patch_size": 8,
                "in_chans": 1,
                "embed_dim": 16,
                "depth": 3,
                "num_heads": 2,
                "n_storage_tokens": 1,
                "layerscale_init": 1e-05,
                "mask_k_bias": True,
                "pos_embed_rope_dtype": "fp32",
                "pos_embed_rope_type": "vanilla",
                "drop_path_rate": 0.0,
                "use_fa4": False,
                "attn_max_score": 100000.0,
            },
        },
        "head": {
            "name": "UNETRHead",
            "kwargs": {
                "in_dim": 16,
                "patch_size": [8, 8, 8],
                "out_channels": None,
                "in_channels": 1,
                "widths": [2, 2, 2],
                "image_skip": True,
                "zero_init_output": False,
            },
        },
        "skip_layers": [0, 1, 2],
        "algorithm": {"name": algorithm, "kwargs": {}},
        "input": {
            "channels": 1,
            "spatial_axes": "xyz",
            "intensity_range": [0.0, 1.0],
            "size_multiple": 8,
            "training_crop": [16, 16, 24],
        },
        "source": {"run": "tiny", "step": 7, "load_with_mia_train_commit": COMMIT},
    }
    if algorithm == "semantic_seg":
        leaf = {"kind": "leaf", "members": None, "tested": True}
        config["classes"] = [
            {
                "channel": 0,
                "name": "unannotated",
                "kind": "unannotated",
                "members": None,
            },
            {**leaf, "channel": 1, "name": "ecs"},
            {**leaf, "channel": 2, "name": "mito_mem"},
            {**leaf, "channel": 3, "name": "mito_lum"},
            {
                "channel": 4,
                "name": "mito",
                "kind": "composite",
                "members": [2, 3],
                "tested": True,
            },
        ]
        config["head"]["kwargs"]["out_channels"] = 5
        config["input"]["voxel_size_nm"] = "native: each dataset's own voxel size"
    else:
        config["offsets"] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]
        config["head"]["kwargs"]["out_channels"] = 6
        config["input"]["voxel_size_nm"] = [8.0, 8.0, 4.0]
    return config


@pytest.fixture
def mia_train():
    pytest.importorskip("models.dinov3_vit3d")
    pytest.importorskip("layers.common.dense_heads")
    pytest.importorskip("safetensors")


def write_source(folder, config):
    """A mia-train model folder with random weights; the network they are."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    net = MiaNetwork(config).eval()
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config))
    weights = {k: v.contiguous() for k, v in net.state_dict().items()}
    save_file(weights, str(folder / "model.safetensors"))
    return net


def network_logits(net, x):
    """The network's logits for x (B, 1, z, y, x), back in z, y, x."""
    with torch.no_grad():
        return net(x.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)


def full_metadata(**serving):
    return ModelMetadata(
        model_name="m",
        model_type="t",
        framework="torch",
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        iteration=1,
        input_voxel_size=[8, 8, 8],
        output_voxel_size=[8, 8, 8],
        channels_names=["a_aff_1_0_0", "a_aff_0_1_0"],
        input_shape=[24, 16, 16],
        output_shape=[24, 16, 16],
        inference_input_shape=[24, 16, 16],
        inference_output_shape=[24, 16, 16],
        author="a",
        description="d",
        **serving,
    )


def test_serving_metadata_round_trips_and_is_never_prompted_for(tmp_path, monkeypatch):
    serving = dict(
        overlap=[6, 4, 4],
        block_phase="corner",
        blend="ramp",
        output_dtype="uint8",
        affinity_offsets=[[1, 0, 0], [0, 1, 0]],
        native_voxel_size=True,
    )
    metadata = full_metadata(**serving)
    write_metadata(metadata, str(tmp_path))
    written = json.loads((tmp_path / "metadata.json").read_text())
    assert {key: written[key] for key in serving} == serving
    assert ModelMetadata(**written) == metadata
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "ramp" in readme and "(0, 1, 0)" in readme

    # A file from before these fields still loads, without them.
    old = ModelMetadata(**{k: v for k, v in written.items() if k not in serving})
    assert all(getattr(old, key) is None for key in serving)
    # ...and an exporter that prompts for what is missing does not ask for them.
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted"))
    assert prompt_for_missing_fields(full_metadata()) == full_metadata()


def test_is_mia_config():
    assert is_mia_config(tiny_config("semantic_seg"))
    assert is_mia_config(tiny_config("affinity_seg"))
    config = tiny_config("affinity_seg")
    # A simmim pretraining repo: the encoder alone.
    assert not is_mia_config({k: config[k] for k in ("model", "input", "source")})
    assert not is_mia_config({**config, "algorithm": {"name": "simmim"}})
    assert not is_mia_config({**config, "model": {"name": "vit3d", "kwargs": {}}})
    assert not is_mia_config({k: v for k, v in config.items() if k != "head"})
    assert not is_mia_config([config])


def test_without_mia_train_it_says_how_to_install(monkeypatch):
    for name in (
        "models",
        "models.dinov3_vit3d",
        "layers",
        "layers.common.dense_heads",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(
        ImportError, match=f'pip install "mia-train @ git\\+.*@{COMMIT}"'
    ):
        MiaNetwork(tiny_config("semantic_seg"))


@pytest.mark.filterwarnings("ignore:No CUDA")
def test_semantic_export(tmp_path, mia_train):
    config = tiny_config("semantic_seg")
    net = write_source(tmp_path / "source", config)
    dinov3 = "license: other\nlicense_name: dinov3-license\nlicense_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license\n"
    (tmp_path / "source" / "README.md").write_text(f"---\n{dinov3}---\n# tiny\n")
    folder = export_mia_model(str(tmp_path / "source"), str(tmp_path / "export"))

    # Its metadata.json is mia_metadata's, made without the weights, but for
    # the device it was traced on, which ends its description.
    written = json.loads((tmp_path / "export" / "metadata.json").read_text())
    description, _ = written.pop("description").split(" Traced on ")
    planned = mia_metadata(str(tmp_path / "source")).model_dump()
    assert {**written, "description": description} == planned

    model = CellmapModel(folder)
    metadata = model.metadata
    # Its card's licence, not cellmap-models' bsd-3-clause, on the card pushed.
    assert f"\n{dinov3}---\n" in generate_huggingface_readme(metadata)
    # The tested classes, a composite one the sum of its leaves.
    assert metadata.channels_names == ["ecs", "mito_mem", "mito_lum", "mito"]
    assert metadata.output_dtype == "uint8"
    # The softmax is in the export, which says so.
    assert metadata.inference_output_activation.model_dump() == {
        "name": "softmax",
        "apply_after_model": False,
        "dim": 1,
    }
    assert (metadata.overlap, metadata.block_phase, metadata.blend) == (
        [6, 4, 4],
        "corner",
        "ramp",
    )
    assert metadata.native_voxel_size is True and metadata.input_voxel_size == [8, 8, 8]
    assert metadata.affinity_offsets is None
    assert (
        metadata.inference_input_shape
        == metadata.inference_output_shape
        == [24, 16, 16]
    )
    assert metadata.iteration == 7

    x = torch.rand(1, 1, 24, 16, 16)
    out = model.ts_model(x)
    assert out.shape == (1, 4, 24, 16, 16) and out.dtype == torch.uint8
    p = torch.softmax(network_logits(net, x), dim=1)
    expected = torch.stack([p[:, 1], p[:, 2], p[:, 3], p[:, 2] + p[:, 3]], dim=1)
    assert (out.float() / 255 - expected).abs().max() <= 0.5 / 255 + 1e-5

    wrapper, metadata = load_mia_model(str(tmp_path / "source"), classes=["mito"])
    assert metadata.channels_names == ["mito"] and wrapper(x).shape == (
        1,
        1,
        24,
        16,
        16,
    )
    with pytest.raises(ValueError, match="no class"):
        load_mia_model(str(tmp_path / "source"), classes=["golgi"])


@pytest.mark.filterwarnings("ignore:No CUDA")
def test_affinity_export(tmp_path, mia_train):
    config = tiny_config("affinity_seg")
    net = write_source(tmp_path / "source", config)
    folder = export_mia_model(str(tmp_path / "source"), str(tmp_path / "export"))

    model = CellmapModel(folder)
    metadata = model.metadata
    # z, y, x order, short range first; the config's are x, y, z.
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert metadata.affinity_offsets == offsets
    assert all("_aff" in name for name in metadata.channels_names)
    assert metadata.output_dtype == "float32"
    # The sigmoid is in the export, which says so.
    assert metadata.inference_output_activation.name == "sigmoid"
    assert not metadata.inference_output_activation.apply_after_model
    assert (
        metadata.input_voxel_size == [4, 8, 8] and metadata.native_voxel_size is False
    )

    x = torch.rand(1, 1, 24, 16, 16)
    out = model.ts_model(x)
    assert out.shape == (1, 6, 24, 16, 16) and out.dtype == torch.float32
    logits = network_logits(net, x)
    for channel, offset in enumerate(offsets):
        network_channel = config["offsets"].index(offset[::-1])
        expected = torch.sigmoid(0.2 * logits[:, network_channel])
        torch.testing.assert_close(out[:, channel], expected)

    if os.path.exists(os.path.join(folder, "model.pt2")):
        torch.testing.assert_close(model.exported_model.module()(x), out)
