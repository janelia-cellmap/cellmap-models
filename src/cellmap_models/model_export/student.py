"""Small students of big 3D networks, to train by distillation into models a browser can run.

``distill_student(teacher, max_parameters, teacher_input, teacher_output, out_channels)`` builds
the student, with at most ``max_parameters`` parameters:

* A funlib U-Net teacher (one ``funlib.learn.torch.models.UNet`` head, alone or followed by one
  final conv, as ``prune_funlib_unet`` covers, e.g. ``fly_organelles.model.StandardUnet``) gets the
  same network with ``num_fmaps * fmap_inc_factor**level`` channels per level for a smaller
  ``num_fmaps`` and ``fmap_inc_factor``, initialized from the teacher's strongest channels by
  ``prune_funlib_unet`` (``init="pruned"``). It is the teacher's own module tree narrowed, so it
  keeps the teacher's pooling, upsampling, activations and block, and it works for a module
  unflattened from ``torch.export`` as for an eager one.
* Any other teacher gets a ``StandardUnet`` (fly_organelles' layout, rebuilt here without funlib)
  with the teacher's channels and output activation, randomly initialized (``init="random"``), at
  the largest block whose output fits inside the teacher's and is centred on the same voxels: both
  networks then place their output in the middle of their input, so the teacher's output, cropped
  to the middle, is the student's target.

Either way it is an initialization: train it to reproduce the teacher's output on raw data (no
labels needed); ``onnx_web.output_difference`` measures how close it got.

The widths: of the ``(num_fmaps, fmap_inc_factor)`` pairs up to the teacher's (StandardUnet's
defaults, 16 and 6, for a random student) whose student fits ``max_parameters``, those of at least
half the largest that fits are candidates, and the one with the most channels at full resolution
wins: the deep levels hold most of the parameters and add context, the top level decides the
output voxel by voxel and costs few. mito-aff-unet-setup-16 (StandardUnet 16/6, 792M parameters)
at 12M gives 13/3, 10.8M parameters.

Example::

    student = distill_student(teacher, 12_000_000, (178,) * 3, (56,) * 3, out_channels=3)
    student.network, student.init, student.input_size, student.output_size
"""

import copy
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from .prune import prune_funlib_unet

# StandardUnet's settings: funlib UNet defaults where fly_organelles' StandardUnet leaves them.
DOWNSAMPLE = (2, 2, 2)
LEVELS = 4
CONVS_DOWN = 3
CONVS_UP = 2  # StandardUnet does not pass its kernel_size_up, so funlib's default
KERNEL = (3, 3, 3)


class ConvPass(torch.nn.Module):
    """funlib's ConvPass: ``convs`` valid convolutions, each followed by a ReLU."""

    def __init__(self, in_channels: int, out_channels: int, convs: int):
        super().__init__()
        layers = []
        for _ in range(convs):
            layers += [
                torch.nn.Conv3d(in_channels, out_channels, KERNEL),
                torch.nn.ReLU(),
            ]
            in_channels = out_channels
        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    """funlib's Downsample: a max pool that refuses a shape it does not divide."""

    def __init__(self):
        super().__init__()
        self.down = torch.nn.MaxPool3d(DOWNSAMPLE, stride=DOWNSAMPLE)

    def forward(self, x):
        if any(s % f for s, f in zip(x.shape[-3:], DOWNSAMPLE)):
            raise RuntimeError(
                f"Can not downsample shape {tuple(x.shape)} by {DOWNSAMPLE}"
            )
        return self.down(x)


def _centre(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Center-crop the spatial axes of ``x`` to ``shape``, as funlib's ``Upsample.crop``."""
    (a, b, c), (sa, sb, sc) = [(n - s) // 2 for n, s in zip(x.shape[-3:], shape)], shape
    return x[..., a : a + sa, b : b + sb, c : c + sc]


class Upsample(torch.nn.Module):
    """funlib's Upsample with ``constant_upsample=True``: nearest upsampling, cropped so that the
    convs after it leave a multiple of ``crop_factor`` (translation equivariance), then the skip,
    cropped to it, concatenated first."""

    def __init__(self, crop_factor: Sequence[int], next_convs: int):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=DOWNSAMPLE, mode="nearest")
        self.crop_factor = tuple(crop_factor)
        self.convolution_crop = next_convs * (KERNEL[0] - 1)

    def forward(self, f_left, g_out):
        g_up = self.up(g_out)
        c = self.convolution_crop
        shape = [
            (s - c) // f * f + c for s, f in zip(g_up.shape[-3:], self.crop_factor)
        ]
        if any(s <= c for s in shape):
            raise RuntimeError(
                f"Feature map {tuple(g_up.shape)} too small for translation equivariance"
            )
        g_cropped = _centre(g_up, shape)
        return torch.cat([_centre(f_left, g_cropped.shape[-3:]), g_cropped], dim=1)


class UNet(torch.nn.Module):
    """funlib's ``UNet`` (one head, valid padding) at StandardUnet's settings."""

    def __init__(self, in_channels: int, num_fmaps: int, fmap_inc_factor: int):
        super().__init__()
        w = [num_fmaps * fmap_inc_factor**level for level in range(LEVELS)]
        self.l_conv = torch.nn.ModuleList(
            [
                ConvPass(in_channels if lv == 0 else w[lv - 1], w[lv], CONVS_DOWN)
                for lv in range(LEVELS)
            ]
        )
        self.l_down = torch.nn.ModuleList([Downsample() for _ in range(LEVELS - 1)])
        # product of the downsample factors below each level
        crop_factors = [
            [f ** (LEVELS - 1 - lv) for f in DOWNSAMPLE] for lv in range(LEVELS - 1)
        ]
        self.r_up = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [Upsample(crop_factors[lv], CONVS_UP) for lv in range(LEVELS - 1)]
                )
            ]
        )
        self.r_conv = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        ConvPass(w[lv] + w[lv + 1], w[lv], CONVS_UP)
                        for lv in range(LEVELS - 1)
                    ]
                )
            ]
        )

    def forward(self, x):
        return self._level(LEVELS - 1, x)

    def _level(self, level, f_in):
        i = LEVELS - level - 1
        f_left = self.l_conv[i](f_in)
        if level == 0:
            return f_left
        g_out = self._level(level - 1, self.l_down[i](f_left))
        return self.r_conv[0][i](self.r_up[0][i](f_left, g_out))


class StandardUnet(torch.nn.Module):
    """``fly_organelles.model.StandardUnet`` without funlib: funlib's UNet with three valid 3x3x3
    convs (+ ReLU) per level going down and two coming up, 2x2x2 max pooling three times, nearest
    upsampling, then a 1x1x1 conv to ``out_channels``, and here an optional output ``activation``
    (e.g. ``torch.nn.Sigmoid()``). Modules are named as theirs, so the state dict loads into
    fly_organelles' StandardUnet (and ``prune_funlib_unet`` reads it). 178^3 in gives 56^3 out.
    """

    def __init__(
        self,
        out_channels: int,
        num_fmaps: int = 16,
        fmap_inc_factor: int = 6,
        in_channels: int = 1,
        activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.unet_backbone = UNet(in_channels, num_fmaps, fmap_inc_factor)
        self.final_conv = torch.nn.Conv3d(num_fmaps, out_channels, (1, 1, 1))
        self.activation = activation if activation is not None else torch.nn.Identity()

    def forward(self, x):
        return self.activation(self.final_conv(self.unet_backbone(x)))


def standard_unet_output_side(side: int) -> Optional[int]:
    """Side of the cube StandardUnet outputs for an input cube of ``side`` voxels, or None for a
    side it cannot take (a level that does not pool to whole voxels, or a feature map too small),
    following its forward: valid convs down, pooling, then upsampling, the crop to the factor below
    and valid convs up."""
    shrink_down, shrink_up = CONVS_DOWN * (KERNEL[0] - 1), CONVS_UP * (KERNEL[0] - 1)
    factor = DOWNSAMPLE[0]
    skips = []
    for level in range(LEVELS):
        side -= shrink_down
        if side <= 0 or (level < LEVELS - 1 and side % factor):
            return None
        skips.append(side)
        if level < LEVELS - 1:
            side //= factor
    for level in reversed(range(LEVELS - 1)):
        crop = factor ** (LEVELS - 1 - level)
        side = (side * factor - shrink_up) // crop * crop + shrink_up
        if side <= shrink_up or side > skips[level]:
            return None
        side -= shrink_up
    return side


def _standard_unet_block(
    teacher_input: Sequence[int], teacher_output: Sequence[int]
) -> Tuple[int, int]:
    """The largest cubic (input, output) StandardUnet block whose output fits in ``teacher_output``
    and is centred on the same voxels: each side differs from the teacher's by an even number of
    voxels."""
    largest = min(teacher_output)
    inputs: Dict[int, int] = {}  # output side -> the smallest input side giving it
    for side in range(
        1, largest + 256
    ):  # StandardUnet reads 122 voxels beyond its output
        out = standard_unet_output_side(side)
        if out is not None and out <= largest:
            inputs.setdefault(out, side)
    for out in sorted(inputs, reverse=True):
        side = inputs[out]
        if all((t - side) % 2 == 0 for t in teacher_input) and all(
            (t - out) % 2 == 0 for t in teacher_output
        ):
            return side, out
    raise ValueError(
        f"no StandardUnet block fits inside the teacher's output {tuple(teacher_output)}, "
        "centred on it"
    )


@dataclass(frozen=True)
class UNetLayout:
    """Where a funlib UNet sits in a state dict: its key ``prefix`` ("unet_backbone." in
    StandardUnet), the ``final_key`` conv after it (or None), the conv indices of each level's
    ``conv_pass`` (``encoder``, ``decoder``), its input channels, its widths, and the channels out of
    the decoder's top level (``outputs``: the network's output without a final conv)."""

    prefix: str
    final_key: Optional[str]
    encoder: tuple
    decoder: tuple
    in_channels: int
    num_fmaps: int
    fmap_inc_factor: int
    outputs: int

    @property
    def levels(self) -> int:
        return len(self.encoder)


def _not_covered(why: str) -> ValueError:
    return ValueError(
        f"{why}: not a funlib UNet with one head, alone or with one final conv"
    )


def _conv_ids(sd, block: str) -> tuple:
    pat = re.compile(re.escape(block) + r"\.conv_pass\.(\d+)\.weight")
    return tuple(sorted(int(m.group(1)) for k in sd for m in [pat.fullmatch(k)] if m))


def unet_layout(sd: Dict[str, torch.Tensor]) -> UNetLayout:
    """The funlib UNet of state dict ``sd`` as ``prune_funlib_unet`` reads it.

    Raises:
        ValueError: ``sd`` is not one it covers: no UNet (or several, or several heads), widths
            that are not ``num_fmaps * fmap_inc_factor**level``, or any tensor besides the UNet's
            and one final conv's, or of another shape than funlib's UNet would give it.
    """
    prefixes = {
        m.group(1)
        for k in sd
        for m in [re.fullmatch(r"(.*)l_conv\.\d+\.conv_pass\.\d+\.weight", k)]
        if m
    }
    if len(prefixes) != 1:
        raise _not_covered(
            "no l_conv.<level>.conv_pass convs"
            if not prefixes
            else f"{len(prefixes)} UNets"
        )
    p = prefixes.pop()
    if any(k.startswith(f"{p}r_conv.1.") for k in sd):
        raise _not_covered("more than one head")
    levels = 0
    while _conv_ids(sd, f"{p}l_conv.{levels}"):
        levels += 1
    encoder = tuple(_conv_ids(sd, f"{p}l_conv.{lv}") for lv in range(levels))
    decoder = tuple(_conv_ids(sd, f"{p}r_conv.0.{lv}") for lv in range(levels - 1))
    if levels < 2 or not all(decoder):
        raise _not_covered(
            f"{levels} encoder level(s) and decoder levels {[lv for lv, c in enumerate(decoder) if c]}"
        )
    widths = [
        int(sd[f"{p}l_conv.{lv}.conv_pass.{encoder[lv][-1]}.weight"].shape[0])
        for lv in range(levels)
    ]
    num_fmaps, factor = widths[0], widths[1] // widths[0]
    if any(w != num_fmaps * factor**lv for lv, w in enumerate(widths)):
        raise _not_covered(
            f"widths {widths} are not num_fmaps * fmap_inc_factor**level"
        )
    others = sorted({k.rpartition(".")[0] for k in sd if not k.startswith(p)})
    if len(others) > 1 or (others and f"{others[0]}.weight" not in sd):
        raise _not_covered(f"layers besides the UNet and one final conv: {others}")
    layout = UNetLayout(
        prefix=p,
        final_key=others[0] if others else None,
        encoder=encoder,
        decoder=decoder,
        in_channels=int(sd[f"{p}l_conv.0.conv_pass.{encoder[0][0]}.weight"].shape[1]),
        num_fmaps=num_fmaps,
        fmap_inc_factor=factor,
        outputs=int(sd[f"{p}r_conv.0.0.conv_pass.{decoder[0][-1]}.weight"].shape[0]),
    )
    # the teacher is the student of its own widths: funlib's shapes at them check the rest
    expected = student_shapes(sd, layout, num_fmaps, factor)
    for k in sorted(set(expected) | set(sd)):
        have = tuple(sd[k].shape) if k in sd else None
        if have != expected.get(k):
            raise _not_covered(
                f"{k} is {have}, where funlib's UNet has {expected.get(k)}"
            )
    return layout


def student_shapes(
    sd, layout: UNetLayout, num_fmaps: int, fmap_inc_factor: int
) -> Dict[str, tuple]:
    """Shape of each tensor of ``sd`` in its UNet at ``num_fmaps * fmap_inc_factor**level`` channels
    per level, as funlib lays them out: the decoder's first conv takes ``cat([skip, upsampled])``, a
    transposed-conv upsampling (``constant_upsample=False``) maps the level below onto itself, and
    the final conv takes the decoder's top level, which keeps its channels without a final conv.
    """
    w = [num_fmaps * fmap_inc_factor**lv for lv in range(layout.levels)]
    top = w[0] if layout.final_key else layout.outputs
    shapes = {}

    def conv(key, cout, cin, transposed=False):
        kernel = tuple(sd[f"{key}.weight"].shape[2:])
        shapes[f"{key}.weight"] = ((cin, cout) if transposed else (cout, cin)) + kernel
        if f"{key}.bias" in sd:
            shapes[f"{key}.bias"] = (cout,)

    def conv_pass(block, ids, cout, cin):
        for j, i in enumerate(ids):
            conv(f"{block}.conv_pass.{i}", cout, cin if j == 0 else cout)

    p = layout.prefix
    for lv, ids in enumerate(layout.encoder):
        conv_pass(
            f"{p}l_conv.{lv}", ids, w[lv], w[lv - 1] if lv else layout.in_channels
        )
    for lv, ids in enumerate(layout.decoder):
        if f"{p}r_up.0.{lv}.up.weight" in sd:
            conv(f"{p}r_up.0.{lv}.up", w[lv + 1], w[lv + 1], transposed=True)
        conv_pass(
            f"{p}r_conv.0.{lv}", ids, top if lv == 0 else w[lv], w[lv] + w[lv + 1]
        )
    if layout.final_key:
        conv(layout.final_key, int(sd[f"{layout.final_key}.weight"].shape[0]), top)
    return shapes


def _count(shapes) -> int:
    return sum(math.prod(s) for s in shapes.values())


def student_widths(sd, layout: UNetLayout, max_parameters: int) -> Tuple[int, int]:
    """The student's ``(num_fmaps, fmap_inc_factor)``: each at most the teacher's (so every level
    keeps a subset of its channels), not the teacher's own, at most ``max_parameters`` parameters;
    of those, the ones of at least half the largest that fits, and among them the most channels at
    full resolution (the module docstring says why)."""
    fitting = {}
    for s in range(1, layout.num_fmaps + 1):
        for g in range(1, layout.fmap_inc_factor + 1):
            n = _count(student_shapes(sd, layout, s, g))
            if n <= max_parameters and (s, g) != (
                layout.num_fmaps,
                layout.fmap_inc_factor,
            ):
                fitting[s, g] = n
    if not fitting:
        raise ValueError(
            f"no narrower copy of the network has at most {max_parameters:,} parameters"
        )
    largest = max(fitting.values())
    return max(
        (sg for sg, n in fitting.items() if n >= largest / 2),
        key=lambda sg: (sg[0], fitting[sg]),
    )


def _narrowed(network: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> torch.nn.Module:
    """A copy of ``network`` with the tensors of ``sd`` (every one of its tensors) and each conv's
    channel counts made its new weight's (onnx_web's rewrites build their layers from them).
    """
    # the copy shares the teacher's tensors until each is replaced, rather than copying them first
    narrow = copy.deepcopy(
        network, memo={id(t): t for t in network.state_dict(keep_vars=True).values()}
    )
    for key, tensor in sd.items():
        path, _, name = key.rpartition(".")
        setattr(narrow.get_submodule(path), name, torch.nn.Parameter(tensor.detach()))
    for m in narrow.modules():
        if isinstance(
            m,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            m.in_channels, m.out_channels = (
                m.weight.shape[0],
                m.weight.shape[1] * m.groups,
            )
        elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            m.in_channels, m.out_channels = (
                m.weight.shape[1] * m.groups,
                m.weight.shape[0],
            )
    return narrow


@dataclass
class Student:
    """A student ``distill_student`` built: the ``network``, ``init`` ("pruned" from the teacher by
    ``prune_funlib_unet``, or "random"), its widths, its parameter count, and the block it computes
    (``input_size``, ``output_size``: voxels per axis)."""

    network: torch.nn.Module
    init: str
    num_fmaps: int
    fmap_inc_factor: int
    parameters: int
    input_size: Tuple[int, ...]
    output_size: Tuple[int, ...]


def distill_student(
    teacher: torch.nn.Module,
    max_parameters: int,
    teacher_input: Sequence[int],
    teacher_output: Sequence[int],
    out_channels: int,
    in_channels: int = 1,
    activation: Optional[torch.nn.Module] = None,
) -> Student:
    """Build a student of ``teacher`` with at most ``max_parameters`` parameters (see the module
    docstring): pruned from it when it is a funlib UNet ``prune_funlib_unet`` covers, else a random
    ``StandardUnet``.

    Args:
        teacher: the trained network (eager, or unflattened from ``torch.export``); not modified.
        max_parameters: the student's parameter budget.
        teacher_input, teacher_output: the teacher's block, voxels per axis: a pruned student
            computes the same, a random one the largest that fits in it, centred on it.
        out_channels, in_channels: the teacher's channels, for a random student.
        activation: the teacher's output activation (e.g. ``torch.nn.Sigmoid()``), ending a random
            student so that its output reads as the teacher's; a pruned one keeps the teacher's.
    """
    sd = teacher.state_dict()
    try:
        layout = unet_layout(sd)
    except ValueError:
        layout = None
    if layout is not None:
        s, g = student_widths(sd, layout, max_parameters)
        shapes = student_shapes(sd, layout, s, g)
        meta = {k: torch.empty(v, device="meta") for k, v in shapes.items()}
        pruned = prune_funlib_unet(
            sd, meta, prefix=layout.prefix, final_key=layout.final_key
        )
        return Student(
            _narrowed(teacher, pruned),
            "pruned",
            s,
            g,
            _count(shapes),
            tuple(teacher_input),
            tuple(teacher_output),
        )

    side, out = _standard_unet_block(teacher_input, teacher_output)
    with torch.device("meta"):
        widest = StandardUnet(out_channels, in_channels=in_channels).state_dict()
    s, g = student_widths(widest, unet_layout(widest), max_parameters)
    network = StandardUnet(out_channels, s, g, in_channels, activation)
    n = sum(p.numel() for p in network.parameters())
    return Student(network, "random", s, g, n, (side,) * 3, (out,) * 3)
