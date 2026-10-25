import copy

import numpy as np
import pytest
import torch

from cellmap_models.model_export.onnx_web import (
    Conv3dAs2d,
    ConvTranspose3dAsConv,
    prepare_for_web,
)


class TinyUNet(torch.nn.Module):
    """Valid-padding 3D conv net with a strided conv and a k=s=2 ConvTranspose3d."""

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv3d(1, 4, 3)
        self.down = torch.nn.Conv3d(4, 6, 2, stride=2)
        self.c2 = torch.nn.Conv3d(6, 6, 3)
        self.up = torch.nn.ConvTranspose3d(6, 4, 2, stride=2)
        self.head = torch.nn.Conv3d(4, 2, 1)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(self.down(x)))
        return self.head(self.up(x))


@pytest.mark.parametrize("k,s", [(3, 1), (2, 2), (1, 1), ((3, 1, 1), 1)])
def test_conv3d_as_2d_exact(k, s):
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(3, 5, k, stride=s)
    x = torch.randn(1, 3, 12, 11, 10)
    with torch.no_grad():
        torch.testing.assert_close(Conv3dAs2d(conv)(x), conv(x), atol=1e-5, rtol=1e-5)


def test_convtranspose3d_as_conv_exact():
    torch.manual_seed(0)
    ct = torch.nn.ConvTranspose3d(3, 4, 2, stride=2)
    x = torch.randn(1, 3, 5, 6, 7)
    with torch.no_grad():
        torch.testing.assert_close(
            ConvTranspose3dAsConv(ct)(x), ct(x), atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("native", [False, True])
def test_prepare_for_web_exact_and_copy(native):
    torch.manual_seed(0)
    model = TinyUNet().eval()
    before = copy.deepcopy(model.state_dict())
    web = prepare_for_web(model, native_conv3d=native)
    assert not any(isinstance(m, torch.nn.ConvTranspose3d) for m in web.modules())
    assert any(isinstance(m, torch.nn.Conv3d) for m in web.modules()) == native
    x = torch.randn(1, 1, 18, 18, 18)
    with torch.no_grad():
        torch.testing.assert_close(web(x), model(x), atol=1e-5, rtol=1e-5)
    # the original model is untouched
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_export_onnx_web(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from cellmap_models.model_export.onnx_web import VARIANTS, export_onnx_web

    torch.manual_seed(0)
    results = export_onnx_web(TinyUNet(), str(tmp_path), "tiny", input_sizes=[18, 22])
    assert len(results) == 2 * len(VARIANTS)
    for r in results:
        assert (tmp_path / r["file"]).exists()
        tol = 1e-4 if r["variant"] == "fp32" else 0.5
        assert r["max_abs"] < tol, r
    assert (tmp_path / "manifest.json").exists()

    export_onnx_web(
        TinyUNet(),
        str(tmp_path),
        "tiny",
        input_sizes=[18],
        variants=["fp32"],
        native_conv3d=True,
    )
    assert (tmp_path / "tiny_in18_conv3d_fp32.onnx").exists()
    assert (tmp_path / "manifest_conv3d.json").exists()
