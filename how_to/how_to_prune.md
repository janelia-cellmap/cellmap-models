# How to Initialize a Small Model by Pruning a Big One

`cellmap_models.model_export.prune.prune_funlib_unet` builds the weights of a smaller U-Net (fewer channels
per level) from a trained big one by **structured L1 channel selection**. It is an initialization for
fine-tuning, much better than random init — not a compression that works without retraining.

Works for any model wrapping one `funlib.learn.torch.models.UNet` (one head) followed by a final conv,
e.g. `fly_organelles.model.StandardUnet`.

## Usage

```python
import torch
from fly_organelles.model import StandardUnet
from cellmap_models.model_export import prune_funlib_unet

teacher = StandardUnet(13, num_fmaps=16, fmap_inc_factor=6)          # 792M params, affinities + LSD
teacher.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])

student = StandardUnet(3, num_fmaps=12, fmap_inc_factor=3)            # 9.2M params, affinities only
sd = prune_funlib_unet(teacher.state_dict(), student.state_dict())   # final conv keeps channels [0:3]
student.load_state_dict(sd)
torch.save({"model_state_dict": sd}, "pretrained_init_checkpoint")   # then fine-tune from it
```

The student's shapes define how many channels are kept; only shapes of `student.state_dict()` are used.
Options: `prefix` (key prefix of the UNet, `"unet_backbone."` for StandardUnet, `""` for a bare UNet),
`final_key` (final conv module name, `None` if absent), `final_out_idx` (which teacher output channels the
student predicts; default the first `C_out`).

## What it does

* **Encoder** `l_conv[level]`: keep the output channels with the largest L1 weight norm of the level's last
  conv; the first conv takes the kept channels of the level above (all input channels at level 0), the
  next convs map the kept set onto itself.
* **Decoder** `r_conv[level]`: same selection; its input is `cat([skip, upsampled])` (skip first, as in
  funlib's `Upsample.forward`), i.e. the kept encoder channels of the level plus the kept channels of the
  bottleneck (deepest decoder level) or of `r_conv[level+1]`, offset by the teacher's skip width.
  Transposed-conv upsampling weights (`constant_upsample=False`) are sliced on the upsampled channels.
* **Final conv**: keeps `final_out_idx` outputs and the kept channels of `r_conv[0]`.

Keys and shapes are checked against the student. Pruning a model to its own widths returns the identical
function (tested).

## A student for distillation, from any teacher

`cellmap_models.model_export.distill_student` picks the student's size and initialization for you, for
any 3D teacher, to train by distillation (match the teacher's outputs on raw data; no labels needed):

```python
from cellmap_models.model_export import distill_student, output_difference

student = distill_student(teacher, max_parameters=12_000_000,
                          teacher_input=(178,) * 3, teacher_output=(56,) * 3, out_channels=3)
student.init            # "pruned" (a funlib UNet teacher) or "random" (any other)
student.network         # train it to reproduce the teacher on student.input_size blocks
student.output_size     # the block it writes, centred in the teacher's output
```

* A funlib UNet teacher (what `prune_funlib_unet` covers, e.g. `StandardUnet`, also unflattened from
  `torch.export`) gets its own module tree with fewer channels, pruned from it. Widths: of the
  `(num_fmaps, fmap_inc_factor)` pairs up to the teacher's that fit the budget, those of at least half
  the largest that fits, and among them the most channels at full resolution (mito-aff-unet-setup-16,
  16/6 and 792M parameters, at 12M: 13/3, 10.8M).
* Any other teacher gets a random `StandardUnet` (fly_organelles' layout, built without funlib; its
  state dict loads into theirs) with the teacher's channels and output `activation`, at the largest
  block whose output fits inside the teacher's, centred on it: crop the teacher's output to the middle
  `student.output_size` for the target.

`output_difference(out, ref, threshold)` is `compare_onnx`'s metrics (max/mean |diff|, mse and the
fraction of `> threshold` decisions that flip) for the student against the teacher.
