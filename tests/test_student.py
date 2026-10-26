import pytest
import torch

from cellmap_models.model_export.student import (
    StandardUnet,
    distill_student,
    standard_unet_output_side,
)


def _params(network):
    return sum(p.numel() for p in network.parameters())


def test_standard_unet_is_fly_organelles_standard_unet():
    model = pytest.importorskip("fly_organelles.model")
    torch.manual_seed(0)
    theirs = model.StandardUnet(3, num_fmaps=2, fmap_inc_factor=2).eval()
    ours = StandardUnet(3, num_fmaps=2, fmap_inc_factor=2).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    x = torch.randn(1, 1, 178, 178, 178)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), theirs(x))


@pytest.mark.parametrize("side", [130, 170, 177, 178, 186])
def test_standard_unet_output_side_is_its_forward(side):
    network = StandardUnet(1, num_fmaps=1, fmap_inc_factor=1)
    try:
        with torch.no_grad():
            out = network(torch.zeros(1, 1, side, side, side)).shape[-1]
    except RuntimeError:
        out = None
    assert standard_unet_output_side(side) == out


def test_a_funlib_unet_teacher_gets_a_pruned_student():
    torch.manual_seed(0)
    teacher = StandardUnet(2, num_fmaps=4, fmap_inc_factor=3).eval()
    budget = _params(teacher) // 10
    student = distill_student(teacher, budget, (178,) * 3, (56,) * 3, out_channels=2)
    assert student.init == "pruned" and student.parameters <= budget
    assert (student.input_size, student.output_size) == ((178,) * 3, (56,) * 3)
    assert student.parameters == _params(student.network)

    # the teacher's own tree, narrowed: a StandardUnet of the student's widths
    narrow = StandardUnet(2, student.num_fmaps, student.fmap_inc_factor).eval()
    narrow.load_state_dict(student.network.state_dict(), strict=True)
    x = torch.randn(1, 1, 178, 178, 178)
    with torch.no_grad():
        torch.testing.assert_close(student.network.eval()(x), narrow(x))
        assert teacher(x).shape == student.network(x).shape
    # the teacher is left as it was
    assert _params(teacher) > 10 * budget - 10


def test_another_teacher_gets_a_random_standard_unet_inside_its_block():
    teacher = torch.nn.Sequential(torch.nn.Conv3d(1, 8, 3), torch.nn.Conv3d(8, 3, 1))
    student = distill_student(
        teacher, 50_000, (148,) * 3, (60,) * 3, 3, activation=torch.nn.Sigmoid()
    )
    assert student.init == "random" and student.parameters <= 50_000
    # the largest StandardUnet output inside 60^3, centred on it
    assert (student.input_size, student.output_size) == ((178,) * 3, (56,) * 3)
    with torch.no_grad():
        y = student.network(torch.randn(1, 1, *student.input_size))
    assert y.shape == (1, 3, 56, 56, 56) and 0 <= y.min() <= y.max() <= 1

    with pytest.raises(ValueError):  # no StandardUnet block fits inside 4^3
        distill_student(teacher, 50_000, (8,) * 3, (4,) * 3, 3)
    with pytest.raises(ValueError):  # no StandardUnet is this small
        distill_student(teacher, 10, (148,) * 3, (60,) * 3, 3)
