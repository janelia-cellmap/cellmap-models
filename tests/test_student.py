import pytest
import torch

from cellmap_models.model_export.student import (
    DOWNSAMPLE_2D,
    KERNEL_2D,
    StandardUnet,
    distill_student,
    standard_unet_output_shape,
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


@pytest.mark.parametrize(
    "shape,two_d",
    [
        ((130,) * 3, False),
        ((177,) * 3, False),
        ((178, 186, 194), False),
        ((1, 306, 306), True),
        ((1, 305, 306), True),
    ],
)
def test_standard_unet_output_shape_is_its_forward(shape, two_d):
    kw = dict(kernel=KERNEL_2D, downsample=DOWNSAMPLE_2D) if two_d else {}
    network = StandardUnet(1, num_fmaps=1, fmap_inc_factor=1, **kw)
    try:
        with torch.no_grad():
            out = tuple(network(torch.zeros(1, 1, *shape)).shape[2:])
    except RuntimeError:
        out = None
    assert standard_unet_output_shape(shape, **kw) == out


def test_a_funlib_unet_teacher_gets_a_pruned_student():
    torch.manual_seed(0)
    teacher = StandardUnet(2, num_fmaps=4, fmap_inc_factor=3).eval()
    budget = _params(teacher) // 10
    student = distill_student(teacher, budget, (178,) * 3, (56,) * 3, out_channels=2)
    assert student.init == "pruned" and student.parameters <= budget
    assert (student.input_size, student.output_size) == ((178,) * 3, (56,) * 3)
    assert student.parameters == _params(student.network)
    assert _params(teacher) > 10 * budget - 10  # the teacher is left as it was

    # laid out as a StandardUnet, so an eager one, which runs at larger blocks too, and whose
    # output there matches its own block's where they overlap
    assert isinstance(student.network, StandardUnet)
    (larger,) = student.blocks([186, 194])
    assert larger == (194,) * 3
    x = torch.randn(1, 1, *larger)
    with torch.no_grad():
        big = student.network.eval()(x)
        own = student.network(x[..., 8:-8, 8:-8, 8:-8])
    assert big.shape[2:] == (72,) * 3
    torch.testing.assert_close(big[..., 8:-8, 8:-8, 8:-8], own, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("funlib", [True, False])
def test_a_2d_teacher_gets_a_2d_student(funlib):
    torch.manual_seed(0)
    if funlib:
        teacher = StandardUnet(2, 4, 3, kernel=KERNEL_2D, downsample=DOWNSAMPLE_2D)
        block = ((1, 306, 306), (1, 184, 184))
    else:
        teacher = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, (1, 3, 3)), torch.nn.Conv3d(8, 2, 1)
        )
        block = ((1, 300, 300), (1, 298, 298))
    student = distill_student(teacher.eval(), 5_000, *block, out_channels=2)
    assert student.init == ("pruned" if funlib else "random")
    assert student.input_size[0] == student.output_size[0] == 1
    assert all(
        w.shape[2] == 1 for w in student.network.state_dict().values() if w.ndim == 5
    )
    with torch.no_grad():
        y = student.network.eval()(torch.randn(1, 1, *student.input_size))
    assert tuple(y.shape[2:]) == student.output_size
    assert student.output_size[1] <= block[1][1]
    # a slice stays a slice in its larger blocks
    assert all(b[0] == 1 and b[1] >= 556 for b in student.blocks([556]))
