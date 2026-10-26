"""ONNX export for in-browser inference (onnxruntime-web, WebGPU execution provider).

Two exact graph rewrites make 3D U-Nets run on onnxruntime-web's WebGPU EP:

* ``ConvTranspose3dAsConv``: 3D ConvTranspose with kernel == stride == 2 has no WebGPU kernel; it is
  rewritten as a 1x1x1 Conv to ``8 * C_out`` channels followed by a 3D depth-to-space.
* ``Conv3dAs2d``: every unpadded Conv3d is rewritten as a sum of Conv2d calls over a depth-folded
  batch, ``out[z] = bias + sum_t Conv2d(in[z * s + t], W[:, :, t])``. Same weights, same output up to
  float rounding. onnxruntime-web runs Conv3d with a naive shader (one thread per output voxel, no
  data reuse) but Conv2d with a tiled implicit GEMM, so the rewrite is ~16x faster in the browser.
  On CPU / CUDA / TensorRT native Conv3d is already fast and the rewrite is slower (more ops), so
  export with ``native_conv3d=True`` for those runtimes.

Each exported file can be written in four variants (float32 input/output in all of them):

* ``fp32``        baseline
* ``fp16``        fp16 weights + compute (halves file and activation memory)
* ``int8w``       int8 per-output-channel conv weights + DequantizeLinear, fp32 compute (~1/4 file)
* ``int8w_fp16``  int8 conv weights, fp16 compute

and is validated with onnxruntime (CPU) against the original PyTorch model: max/mean |output diff|
and the fraction of voxels whose ``output > 0`` decision flips.

Requires ``pip install cellmap-models[export]`` (onnx, onnxruntime).
"""

import copy
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

VARIANTS = ("fp32", "fp16", "int8w", "int8w_fp16")
OPSET = 18


class ConvTranspose3dAsConv(torch.nn.Module):
    """Exact replacement for ConvTranspose3d with kernel == stride (no overlap; e.g. 2x2x2, or 1x2x2
    in a 2D U-Net): a 1x1x1 Conv to Cout*kz*ky*kx channels followed by a 3D depth-to-space done as
    three rank-6 reshape/transpose steps. onnxruntime-web's WebGPU EP only implements 1D/2D
    ConvTranspose."""

    def __init__(self, ct: torch.nn.ConvTranspose3d):
        super().__init__()
        assert (
            tuple(ct.kernel_size) == tuple(ct.stride)
            and tuple(ct.padding) == (0, 0, 0)
            and tuple(ct.output_padding) == (0, 0, 0)
            and tuple(ct.dilation) == (1, 1, 1)
            and ct.groups == 1
        ), "only ConvTranspose3d(kernel=stride, padding=0, groups=1) is supported"
        cin, cout = ct.in_channels, ct.out_channels
        self.cout = cout
        self.factors = tuple(int(k) for k in ct.kernel_size)
        taps = self.factors[0] * self.factors[1] * self.factors[2]
        self.conv = torch.nn.Conv3d(cin, cout * taps, 1, device=ct.weight.device)
        # ConvTranspose weight is (Cin, Cout, a, b, c); new conv output channel order is (co, a, b, c).
        w = ct.weight.detach().permute(1, 2, 3, 4, 0).reshape(cout * taps, cin, 1, 1, 1)
        self.conv.weight.data.copy_(w)
        if ct.bias is not None:
            self.conv.bias.data.copy_(ct.bias.detach().repeat_interleave(taps))
        else:
            self.conv.bias.data.zero_()

    def forward(self, x):
        y = self.conv(x)
        n, _, d, h, w = y.shape
        co = self.cout
        fz, fy, fx = self.factors
        y = (
            y.reshape(n, co * fz * fy, fx, d, h, w)
            .permute(0, 1, 3, 4, 5, 2)
            .reshape(n, co * fz * fy, d, h, fx * w)
        )
        y = (
            y.reshape(n, co * fz, fy, d, h, fx * w)
            .permute(0, 1, 3, 4, 2, 5)
            .reshape(n, co * fz, d, fy * h, fx * w)
        )
        y = (
            y.reshape(n, co, fz, d, fy * h, fx * w)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, co, fz * d, fy * h, fx * w)
        )
        return y


class Conv3dAs2d(torch.nn.Module):
    """Exact rewrite of an unpadded Conv3d (batch 1) as Conv2d calls with depth folded into the batch
    axis: ``y = sum_t Conv2d(x[t :: stride], W[:, :, t])``; the bias is added once (on tap 0).
    """

    def __init__(self, conv: torch.nn.Conv3d):
        super().__init__()
        assert (
            tuple(conv.padding) == (0, 0, 0)
            and tuple(conv.dilation) == (1, 1, 1)
            and conv.groups == 1
        ), "only unpadded, undilated, ungrouped Conv3d is supported"
        self.kd, self.sd = conv.kernel_size[0], conv.stride[0]
        self.convs = torch.nn.ModuleList()
        for t in range(self.kd):
            with_bias = t == 0 and conv.bias is not None
            c2 = torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size[1:],
                stride=conv.stride[1:],
                bias=with_bias,
                device=conv.weight.device,
            )
            c2.weight.data.copy_(conv.weight.detach()[:, :, t])
            if with_bias:
                c2.bias.data.copy_(conv.bias.detach())
            self.convs.append(c2)

    def forward(self, x):
        xb = x[0].transpose(0, 1)  # [1, C, D, H, W] -> [D, C, H, W]
        d_out = (xb.shape[0] - self.kd) // self.sd + 1
        y = None
        for t, c2 in enumerate(self.convs):
            yt = c2(xb[t : t + self.sd * (d_out - 1) + 1 : self.sd])
            y = yt if y is None else y + yt
        return y.transpose(0, 1).unsqueeze(0)  # [D', Co, H', W'] -> [1, Co, D', H', W']


def conv3d_as_2d(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every Conv3d (in place) with its exact Conv2d-over-depth-batch rewrite."""
    for name, mod in list(model.named_children()):
        if isinstance(mod, torch.nn.Conv3d):
            setattr(model, name, Conv3dAs2d(mod))
        else:
            conv3d_as_2d(mod)
    return model


def webgpu_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """Swap (in place) ops the WebGPU EP can't run for exact equivalents (3D ConvTranspose)."""
    for name, mod in list(model.named_children()):
        if isinstance(mod, torch.nn.ConvTranspose3d):
            setattr(model, name, ConvTranspose3dAsConv(mod))
        else:
            webgpu_compatible(mod)
    return model


def prepare_for_web(
    model: torch.nn.Module, native_conv3d: bool = False
) -> torch.nn.Module:
    """Return a rewritten copy of ``model`` for onnxruntime-web; ``model`` itself is not modified."""
    m = webgpu_compatible(copy.deepcopy(model))
    if not native_conv3d:
        m = conv3d_as_2d(m)
    return m.eval()


def quantize_conv_weights_int8(model):
    """Replace each Conv/ConvTranspose weight initializer (fp32 or fp16) of an ONNX model with int8 +
    per-output-channel symmetric DequantizeLinear."""
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes = []
    for node in g.node:
        if node.op_type in ("Conv", "ConvTranspose") and node.input[1] in inits:
            wname = node.input[1]
            w_orig = numpy_helper.to_array(inits[wname])
            is_fp16 = w_orig.dtype == np.float16
            w = w_orig.astype(np.float32)
            axis = (
                0 if node.op_type == "Conv" else 1
            )  # ConvTranspose weight is (Cin, Cout, k...)
            red = tuple(i for i in range(w.ndim) if i != axis)
            scale = np.maximum(np.abs(w).max(axis=red), 1e-12) / 127.0
            bshape = [1] * w.ndim
            bshape[axis] = -1
            q = np.clip(np.round(w / scale.reshape(bshape)), -127, 127).astype(np.int8)
            g.initializer.remove(inits[wname])
            g.initializer.extend(
                [
                    numpy_helper.from_array(q, wname + "_q"),
                    numpy_helper.from_array(scale.astype(np.float32), wname + "_scale"),
                    numpy_helper.from_array(
                        np.zeros_like(scale, dtype=np.int8), wname + "_zp"
                    ),
                ]
            )
            # DequantizeLinear only emits float32 at opset 18, so fp16 graphs get an extra Cast.
            dq_out = wname + "_f32" if is_fp16 else wname
            new_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [wname + "_q", wname + "_scale", wname + "_zp"],
                    [dq_out],
                    name=wname + "_dq",
                    axis=axis,
                )
            )
            if is_fp16:
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [dq_out],
                        [wname],
                        name=wname + "_cast",
                        to=onnx.TensorProto.FLOAT16,
                    )
                )
    # DequantizeLinear nodes go first so the graph stays topologically sorted.
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return model


def to_fp16(model):
    """fp16 weights and compute, float32 graph inputs/outputs kept."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    return convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False
    )


def export_fp32(model: torch.nn.Module, input_shape: Sequence[int], path: str):
    """Fixed-shape fp32 ONNX export (opset 18, legacy TorchScript exporter) + onnx checker."""
    import onnx

    dummy = torch.zeros(*input_shape)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(path),
            opset_version=OPSET,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamo=False,
        )
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    return m


def output_difference(out: np.ndarray, ref: np.ndarray, threshold: float = 0.0) -> Dict:
    """How far ``out`` is from the reference output ``ref``: max/mean |diff|, the mean squared
    diff, and the fraction of voxels whose ``> threshold`` decision flips (0 for logits, 0.5 for
    probabilities)."""
    diff = np.abs(out - ref)
    return dict(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        mse=float(np.mean(diff**2)),
        flip_frac=float(np.mean((out > threshold) != (ref > threshold))),
    )


def compare_onnx(onnx_path: str, x: np.ndarray, ref: np.ndarray) -> Dict:
    """Run ``onnx_path`` with onnxruntime CPU on ``x`` and compare with the reference output ``ref``."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out = sess.run(None, {"input": x})[0].astype(np.float32)
    dt = time.time() - t0
    assert out.shape == ref.shape, (out.shape, ref.shape)
    return dict(**output_difference(out, ref), ort_cpu_s=round(dt, 3))


def _cpu_copy(model: torch.nn.Module) -> torch.nn.Module:
    """``model`` (on any device) copied to the CPU, its tensors copied straight there rather than
    first beside themselves on their device."""
    memo = {
        id(t): (
            torch.nn.Parameter(t.detach().cpu(), t.requires_grad)
            if isinstance(t, torch.nn.Parameter)
            else t.detach().cpu()
        )
        for t in [*model.parameters(), *model.buffers()]
    }
    return copy.deepcopy(model, memo)


def _block_name(block: Sequence[int]) -> str:
    """A block in a file name, as cellmap/onnx-web-models names them: ``in172`` for a cube,
    ``2d_in556`` for a square slice, else ``in{z}x{y}x{x}``."""
    z, y, x = block
    if z == y == x:
        return f"in{z}"
    if z == 1 and y == x:
        return f"2d_in{y}"
    return f"in{z}x{y}x{x}"


def export_onnx_web(
    model: torch.nn.Module,
    out_dir: str,
    name: str,
    input_sizes: Sequence[Union[int, Sequence[int]]],
    in_channels: int = 1,
    variants: Sequence[str] = VARIANTS,
    native_conv3d: bool = False,
    validation_input: Optional[np.ndarray] = None,
    manifest_extra: Optional[Dict] = None,
) -> List[Dict]:
    """Export a 3D model (input ``[1, C, z, y, x]``) to fixed-size ONNX files for onnxruntime-web.

    For every block in ``input_sizes`` and every variant in ``variants`` writes
    ``{out_dir}/{name}_in{size}[_conv3d]_{variant}.onnx`` (``{name}_2d_in{y}`` for a ``(1, y, y)``
    slice, ``{name}_in{z}x{y}x{x}`` for another shape) and validates it against ``model`` with
    onnxruntime CPU. Writes ``{out_dir}/manifest.json`` (``manifest_conv3d.json`` with ``native_conv3d``)
    with one entry per file and returns the entries.

    Args:
        model: the original PyTorch model (used as the reference, on its own device; not
            modified).
        out_dir: output folder.
        name: file name stem, e.g. ``"setup_53_mito_affinities_16nm"``.
        input_sizes: input blocks, each a cube's side or a ``(z, y, x)`` shape (``(1, y, x)`` for a
            2D model, input ``[1, C, 1, y, x]``); each must be a valid input of the model. The
            manifest says a cube's sizes as one number and any other block's as three.
        in_channels: number of input channels.
        variants: subset of ``VARIANTS``.
        native_conv3d: keep native Conv3d nodes (for CPU/CUDA runtimes) instead of the Conv2d rewrite;
            adds ``_conv3d`` to the file names.
        validation_input: normalized input ``[1, C, z, y, x]`` at least as large as every block
            on every axis, used for validation (centre crops are taken per block). Use real data: random noise makes
            the decision-flip metric meaningless. Defaults to uniform noise in [-1, 1].
        manifest_extra: extra fields stored at the top level of ``manifest.json``.
    """
    import onnx

    unknown = set(variants) - set(VARIANTS)
    assert not unknown, f"unknown variants {unknown}; choose from {VARIANTS}"
    model = model.eval()
    # the reference runs where the model is (a GPU's, say); the export is of a CPU copy
    device = next(model.parameters()).device
    export_model = prepare_for_web(_cpu_copy(model), native_conv3d=native_conv3d)
    os.makedirs(out_dir, exist_ok=True)
    blocks = [
        (b,) * 3 if isinstance(b, int) else tuple(int(n) for n in b)
        for b in input_sizes
    ]
    largest = tuple(max(b[i] for b in blocks) for i in range(3))
    if validation_input is None:
        rng = np.random.default_rng(0)
        validation_input = rng.uniform(-1, 1, (1, in_channels, *largest)).astype(
            np.float32
        )
    assert validation_input.ndim == 5 and all(
        s >= n for s, n in zip(validation_input.shape[2:], largest)
    )

    results = []
    for block in blocks:
        offs = [(s - n) // 2 for s, n in zip(validation_input.shape[2:], block)]
        x = np.ascontiguousarray(
            validation_input[
                :,
                :,
                offs[0] : offs[0] + block[0],
                offs[1] : offs[1] + block[1],
                offs[2] : offs[2] + block[2],
            ],
            dtype=np.float32,
        )
        # float32 without TF32, which would round the reference to ~1e-3
        with (
            torch.no_grad(),
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled, allow_tf32=False
            ),
        ):
            ref = model(torch.from_numpy(x).to(device)).cpu().numpy()
        stem = f"{name}_{_block_name(block)}" + ("_conv3d" if native_conv3d else "")
        fp32_path = os.path.join(out_dir, f"{stem}_fp32.onnx")
        export_fp32(export_model, (1, in_channels, *block), fp32_path)
        cube = len(set(block)) == 1
        for v in variants:
            path = os.path.join(out_dir, f"{stem}_{v}.onnx")
            if v == "fp16":
                onnx.save(to_fp16(onnx.load(fp32_path)), path)
            elif v == "int8w":
                onnx.save(quantize_conv_weights_int8(onnx.load(fp32_path)), path)
            elif v == "int8w_fp16":
                onnx.save(
                    quantize_conv_weights_int8(to_fp16(onnx.load(fp32_path))), path
                )
            r = dict(
                variant=v,
                input_size=block[0] if cube else list(block),
                output_size=(
                    int(ref.shape[2]) if cube else [int(n) for n in ref.shape[2:]]
                ),
                file=os.path.basename(path),
                file_mb=round(os.path.getsize(path) / 1e6, 2),
                conv3d_as_2d=not native_conv3d,
                **compare_onnx(path, x, ref),
            )
            results.append(r)
            print(json.dumps(r), flush=True)
        if "fp32" not in variants:
            os.remove(fp32_path)

    manifest = dict(
        name=name,
        params=sum(p.numel() for p in model.parameters()),
        io="input 'input' float32 [1,C,z,y,x]; output 'output' float32 [1,C',z',y',x']",
        opset=OPSET,
        **(manifest_extra or {}),
        variants=results,
    )
    manifest_name = "manifest_conv3d.json" if native_conv3d else "manifest.json"
    with open(os.path.join(out_dir, manifest_name), "w") as f:
        json.dump(manifest, f, indent=2)
    return results
