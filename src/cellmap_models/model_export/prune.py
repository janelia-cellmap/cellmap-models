"""Structured (channel) pruning of a funlib U-Net to initialize a smaller student from a big teacher.

At every level the student keeps the teacher's output channels with the largest L1 weight norm (ranked
on the level's last conv) and every weight tensor is sliced down to the student's shape, respecting how
channels flow through the U-Net:

* encoder ``l_conv[level]``: input = kept channels of the level above (all input channels at level 0);
  the 2nd/3rd convs of a level map the kept channel set onto itself.
* decoder ``r_conv[level]``: input = ``cat([skip, upsampled])`` (skip first, as in funlib's
  ``Upsample.forward``) where skip = kept encoder channels of that level and upsampled = kept channels of
  the bottleneck (deepest decoder level) or of ``r_conv[level + 1]`` (other levels).
* ``r_up`` transposed convs (``constant_upsample=False``) are sliced on the upsampled channel set.
* the final 1x1 conv keeps the requested output channels (default: the first ``C_out`` of the teacher,
  e.g. affinities ``[0:3]`` of an affinities+LSD head) and the kept channels of ``r_conv[0]``.

This gives fine-tuning a much better start than random init; it is not a drop-in compression (the
student must be fine-tuned). Works for any model wrapping ``funlib.learn.torch.models.UNet`` (one head)
followed by a final conv, e.g. ``fly_organelles.model.StandardUnet``.

Example (``fly_organelles`` StandardUnet, 792M -> 9.2M params)::

    teacher = StandardUnet(13, num_fmaps=16, fmap_inc_factor=6)
    teacher.load_state_dict(torch.load(ckpt)["model_state_dict"])
    student = StandardUnet(3, num_fmaps=12, fmap_inc_factor=3)
    sd = prune_funlib_unet(teacher.state_dict(), student.state_dict())
    student.load_state_dict(sd)
"""

import re
from typing import Dict, Optional, Sequence

import torch


def top_k_by_l1(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted indices of the ``k`` output channels (dim 0) of ``weight`` with the largest L1 norm."""
    l1 = weight.detach().abs().flatten(1).sum(1)
    return torch.argsort(l1, descending=True)[:k].sort().values


def _conv_ids(sd: Dict[str, torch.Tensor], block: str):
    """Sorted Sequential indices of the convs inside ``{block}.conv_pass``."""
    pat = re.compile(re.escape(block) + r"\.conv_pass\.(\d+)\.weight$")
    ids = sorted(int(m.group(1)) for k in sd for m in [pat.match(k)] if m)
    assert ids, f"no convs found under {block}.conv_pass"
    return ids


def _slice(sd, out_sd, key, out_idx, in_idx, transposed=False):
    w = sd[key + ".weight"]
    if transposed:  # ConvTranspose weight is (Cin, Cout, k...)
        out_sd[key + ".weight"] = w[in_idx][:, out_idx].clone()
    else:
        out_sd[key + ".weight"] = w[out_idx][:, in_idx].clone()
    if key + ".bias" in sd:
        out_sd[key + ".bias"] = sd[key + ".bias"][out_idx].clone()


def prune_funlib_unet(
    teacher_sd: Dict[str, torch.Tensor],
    student_sd: Dict[str, torch.Tensor],
    prefix: str = "unet_backbone.",
    final_key: Optional[str] = "final_conv",
    final_out_idx: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Build a student state dict by L1 channel selection from a teacher state dict.

    Args:
        teacher_sd: state dict of the trained teacher.
        student_sd: state dict of a freshly built student of the same architecture family with fewer
            channels per level; only its shapes are used (they define how many channels to keep).
        prefix: key prefix of the funlib UNet inside the model (``"unet_backbone."`` for
            ``StandardUnet``, ``""`` for a bare UNet).
        final_key: module name of the final conv after the UNet (``None`` if there is none).
        final_out_idx: teacher output channels kept by the final conv; defaults to the first ``C_out``.

    Returns:
        a state dict with exactly the student's keys and shapes (checked).
    """
    p = prefix
    levels = sorted(
        {
            int(m.group(1))
            for k in teacher_sd
            for m in [re.match(re.escape(p) + r"l_conv\.(\d+)\.", k)]
            if m
        }
    )
    n_levels = len(levels)
    assert (
        levels == list(range(n_levels)) and n_levels >= 2
    ), f"no funlib UNet found under prefix {p!r}"
    assert not any(
        k.startswith(p + "r_conv.1.") for k in teacher_sd
    ), "only num_heads=1 is supported"
    new = {}

    # encoder: keep top-L1 output channels of each level's last conv
    kept_enc, t_width = [], []
    for lv in range(n_levels):
        blk = f"{p}l_conv.{lv}"
        ids = _conv_ids(teacher_sd, blk)
        last = f"{blk}.conv_pass.{ids[-1]}"
        k = student_sd[last + ".weight"].shape[0]
        t_width.append(teacher_sd[last + ".weight"].shape[0])
        out_idx = top_k_by_l1(teacher_sd[last + ".weight"], k)
        kept_enc.append(out_idx)
        in_idx = (
            torch.arange(teacher_sd[f"{blk}.conv_pass.{ids[0]}.weight"].shape[1])
            if lv == 0
            else kept_enc[lv - 1]
        )
        for j, i in enumerate(ids):
            _slice(
                teacher_sd,
                new,
                f"{blk}.conv_pass.{i}",
                out_idx,
                in_idx if j == 0 else out_idx,
            )

    # decoder, deepest first: input is cat([skip, upsampled])
    kept_dec = {}
    for lv in reversed(range(n_levels - 1)):
        blk = f"{p}r_conv.0.{lv}"
        ids = _conv_ids(teacher_sd, blk)
        last = f"{blk}.conv_pass.{ids[-1]}"
        k = student_sd[last + ".weight"].shape[0]
        out_idx = top_k_by_l1(teacher_sd[last + ".weight"], k)
        kept_dec[lv] = out_idx
        up_idx = kept_enc[lv + 1] if lv == n_levels - 2 else kept_dec[lv + 1]
        up_key = f"{p}r_up.0.{lv}.up"
        if up_key + ".weight" in teacher_sd:  # transposed-conv upsampling: Cup -> Cup
            _slice(teacher_sd, new, up_key, up_idx, up_idx, transposed=True)
        in_idx = torch.cat([kept_enc[lv], up_idx + t_width[lv]])
        for j, i in enumerate(ids):
            _slice(
                teacher_sd,
                new,
                f"{blk}.conv_pass.{i}",
                out_idx,
                in_idx if j == 0 else out_idx,
            )

    if final_key is not None:
        k = student_sd[final_key + ".weight"].shape[0]
        idx = (
            torch.arange(k) if final_out_idx is None else torch.as_tensor(final_out_idx)
        )
        assert (
            len(idx) == k
        ), f"final_out_idx has {len(idx)} channels, student final conv has {k}"
        _slice(teacher_sd, new, final_key, idx, kept_dec[0])

    missing = set(student_sd) - set(new)
    extra = set(new) - set(student_sd)
    assert (
        not missing and not extra
    ), f"key mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}"
    for k, v in student_sd.items():
        assert (
            new[k].shape == v.shape
        ), f"shape mismatch at {k}: {tuple(new[k].shape)} vs {tuple(v.shape)}"
    return new
