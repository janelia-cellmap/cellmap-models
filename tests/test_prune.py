import pytest
import torch

from cellmap_models.model_export.prune import prune_funlib_unet, top_k_by_l1


def test_top_k_by_l1():
    w = torch.zeros(5, 2, 3, 3, 3)
    for i, v in enumerate([0.1, 5.0, 0.2, 3.0, 4.0]):
        w[i] = v
    assert top_k_by_l1(w, 3).tolist() == [1, 3, 4]


class WrappedUNet(torch.nn.Module):
    """Same layout as fly_organelles.model.StandardUnet: funlib UNet + final 1x1 conv."""

    def __init__(
        self, out_channels, num_fmaps, fmap_inc_factor, constant_upsample=True
    ):
        super().__init__()
        from funlib.learn.torch.models import UNet

        self.unet_backbone = UNet(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=[(2, 2, 2), (2, 2, 2)],
            kernel_size_down=[[(3, 3, 3)] * 3] * 3,
            constant_upsample=constant_upsample,
        )
        self.final_conv = torch.nn.Conv3d(num_fmaps, out_channels, 1)

    def forward(self, x):
        return self.final_conv(self.unet_backbone(x))


@pytest.mark.parametrize("constant_upsample", [True, False])
def test_prune_funlib_unet(constant_upsample):
    pytest.importorskip("funlib.learn.torch")
    torch.manual_seed(0)
    teacher = WrappedUNet(5, 4, 3, constant_upsample).eval()
    x = torch.randn(1, 1, 66, 66, 66)

    # pruning to the same widths keeps every channel -> the same function
    same = WrappedUNet(5, 4, 3, constant_upsample)
    same.load_state_dict(prune_funlib_unet(teacher.state_dict(), same.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(same.eval()(x), teacher(x))

    # smaller student: shapes match and it runs
    student = WrappedUNet(2, 3, 2, constant_upsample)
    sd = prune_funlib_unet(
        teacher.state_dict(), student.state_dict(), final_out_idx=[0, 3]
    )
    student.load_state_dict(sd, strict=True)
    torch.testing.assert_close(
        sd["final_conv.bias"], teacher.state_dict()["final_conv.bias"][[0, 3]]
    )
    with torch.no_grad():
        assert student.eval()(x).shape[1] == 2
