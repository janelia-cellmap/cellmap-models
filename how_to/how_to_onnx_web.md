# How to Export a 3D Model for the Browser (onnxruntime-web / WebGPU)

`cellmap_models.model_export.onnx_web` exports a 3D segmentation model to fixed-size ONNX files that run
in the browser with [onnxruntime-web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
on its WebGPU execution provider (used by the cellmap-flow browser app).

## Prerequisites

```bash
pip install -e ".[export,huggingface]"   # onnx, onnxruntime, huggingface-hub
```

## Export

```python
from cellmap_models.model_export import export_onnx_web

model = ...          # torch.nn.Module, input [1, C, z, y, x], valid (unpadded) convolutions
x = ...              # real, normalized EM: np.float32 [1, C, n, n, n] with n >= max(input_sizes)

export_onnx_web(
    model,
    out_dir="export/onnx_web/setup_53",
    name="setup_53_mito_affinities_16nm",
    input_sizes=[172, 140, 124],     # cubic block sizes the model accepts
    validation_input=x,
)
```

This writes, per block size, four files plus `manifest.json`:

| variant | weights / compute | file size |
|---|---|---|
| `fp32` | fp32 / fp32 | 1× |
| `fp16` | fp16 / fp16 | ½ |
| `int8w` | int8 per-channel conv weights / fp32 | ¼ |
| `int8w_fp16` | int8 per-channel conv weights / fp16 | ¼ |

Inputs and outputs stay float32 in all of them. Every file is run with onnxruntime (CPU) and compared
with the original PyTorch model on `validation_input`; `manifest.json` records `max_abs` / `mean_abs`
(output difference) and `flip_frac` (fraction of voxels whose `output > 0` decision changes). Pass real
data — on random noise the decision-flip metric means nothing.

For runtimes with a fast 3D convolution (onnxruntime CPU/CUDA, TensorRT) export a second copy with
native Conv3d nodes (file names get `_conv3d`, manifest `manifest_conv3d.json`):

```python
export_onnx_web(model, "export/onnx_web/conv3d/setup_53", "setup_53_mito_affinities_16nm",
                [172, 140, 124], validation_input=x, native_conv3d=True)
```

## What the export changes, and why

Two exact graph rewrites are applied to a copy of the model (`prepare_for_web`); the original model is
the validation reference.

**ConvTranspose3d → Conv + depth-to-space** (`ConvTranspose3dAsConv`). onnxruntime-web's WebGPU EP has no
3D ConvTranspose. With kernel = stride = 2 the 2×2×2 output blocks do not overlap, so it equals a 1×1×1
conv to `8·C_out` channels followed by scattering those channels into the 2×2×2 neighbourhood.

**Conv3d → sum of Conv2d** (`Conv3dAs2d`, skipped with `native_conv3d=True`). Pull the depth sum out of a
3D convolution:

```
out[z] = bias + Σ_dz Σ_dy,dx W[dz,dy,dx] · in[z+dz, y+dy, x+dx]
       = bias + Conv2d(in[z], W[:,:,0]) + Conv2d(in[z+1], W[:,:,1]) + Conv2d(in[z+2], W[:,:,2])
```

(with depth stride `s`: `out[z] = bias + Σ_t Conv2d(in[z·s + t], W[:,:,t])`). Only the order of the
additions changes, so the output is identical up to float rounding. In the graph the depth axis is folded
into the batch (`[1,C,D,H,W] → [D,C,H,W]`), each depth tap is one Conv2d on a shifted slice range, the taps
are added and the result reshaped back to 5D.

The multiply-adds are the same. The speed-up comes from how onnxruntime-web executes them: Conv3d runs as a
naive shader (one thread per output voxel, every input and weight read from global memory, no reuse →
memory-bound), Conv2d as a tiled implicit GEMM with shared-memory tiles that reuse every loaded value.
Measured in Chrome on an Apple GPU, a 148³ block of a 3D ResUNet went from **20.9 s (native Conv3d) to
1.2 s (rewrite), ~17×**; fp16 does not rescue the naive shader (17–25 s).

This is a WebGPU effect, not a general one: the rewrite adds ops (Slice/Add, BatchNorm no longer folded),
and on onnxruntime CPU it was ~2× *slower* than native Conv3d. Publish both copies and use the rewrite only
in the browser.

**Limits** (asserted): Conv3d must be unpadded, undilated, ungrouped and run with batch 1;
ConvTranspose3d must have kernel = stride = 2, no padding. Valid-padding U-Nets satisfy this.

## Publish

The cellmap-flow browser app reads from the Hugging Face repo `cellmap/onnx-web-models`:
`setup_XX/` holds the rewrite (recommended in the browser), `conv3d/setup_XX/` the native Conv3d copy.

```python
from huggingface_hub import HfApi

HfApi().upload_folder(folder_path="export/onnx_web", repo_id="cellmap/onnx-web-models",
                      repo_type="model", commit_message="Add setup_53")
```
