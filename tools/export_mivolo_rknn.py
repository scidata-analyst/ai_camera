"""Export the local MiVOLO v2 checkpoint to an RK3576 RKNN model.

The input is a single NCHW tensor with six channels: normalized face RGB in
channels 0:3 followed by normalized body RGB in channels 3:6.  The output is
the three raw MiVOLO values: two gender logits and normalized age.

Run on an x86_64 Linux PC with RKNN-Toolkit2 installed:
    python tools/export_mivolo_rknn.py
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class MiVOLOExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, six_channel_input):
        # Keep only the raw three-value output. Age scaling and gender softmax
        # stay in Python, matching the existing app's output handling.
        return self.model.mivolo(
            concat_input=six_channel_input, return_dict=False
        )[0]


def patch_outlook_attention() -> None:
    """Replace Unfold/Fold with grouped convolutions for RKNN export.

    MiVOLO's VOLO attention uses im2col/col2im. RKNN Toolkit2 does not list
    ONNX Col2Im as supported, so express the fixed 3x3 patch transforms with
    grouped Conv2d/ConvTranspose2d instead.
    """
    import torch.nn.functional as F
    from timm.models.volo import OutlookAttention

    def forward(self, x):
        batch, height, width, channels = x.shape
        kernel = self.kernel_size
        h = math.ceil(height / self.stride)
        w = math.ceil(width / self.stride)
        weights = self._rknn_patch_weights.to(dtype=x.dtype)

        value = self.v(x).permute(0, 3, 1, 2)
        value = F.conv2d(value, weights, stride=self.stride,
                         padding=self.padding, groups=channels)
        value = value.reshape(
            batch, self.num_heads, channels // self.num_heads,
            kernel * kernel, h * w,
        ).permute(0, 1, 4, 3, 2)

        attention = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attention = self.attn(attention).reshape(
            batch, h * w, self.num_heads, kernel * kernel,
            kernel * kernel,
        ).permute(0, 2, 1, 3, 4)
        attention = self.attn_drop(attention * self.scale)

        patches = (attention @ value).permute(0, 1, 4, 3, 2).reshape(
            batch, channels * kernel * kernel, h * w,
        )
        patches = patches.reshape(batch, channels * kernel * kernel, h, w)
        base_h = (h - 1) * self.stride - 2 * self.padding + kernel
        base_w = (w - 1) * self.stride - 2 * self.padding + kernel
        out_pad = (height - base_h, width - base_w)
        image = F.conv_transpose2d(
            patches, weights, stride=self.stride, padding=self.padding,
            output_padding=out_pad, groups=channels,
        )
        image = self.proj(image.permute(0, 2, 3, 1))
        return self.proj_drop(image)

    OutlookAttention.forward = forward


def install_patch_weights(model) -> None:
    from timm.models.volo import OutlookAttention

    for module in model.modules():
        if isinstance(module, OutlookAttention):
            kernel = module.kernel_size
            weights = torch.eye(kernel * kernel, dtype=torch.float32).reshape(
                kernel * kernel, 1, kernel, kernel,
            )
            weights = weights.repeat(module.v.out_features, 1, 1, 1)
            module.register_buffer("_rknn_patch_weights", weights, persistent=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default=str(ROOT / "models/mivolo_v2"),
        help="Local Hugging Face MiVOLO v2 snapshot directory",
    )
    parser.add_argument(
        "--onnx", default=str(ROOT / "models/.export_tmp/mivolo_v2_rk3576.onnx"),
    )
    parser.add_argument(
        "--out", default=str(ROOT / "models/mivolo_v2_rk3576.rknn"),
    )
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--keep-onnx", action="store_true")
    args = parser.parse_args()

    # Keep Transformers' dynamic-code cache with the model assets/exports.
    hf_cache = ROOT / "models/huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_cache / "modules"))

    # Apply the compatibility shim used by app/mivolo.py before constructing
    # the remote-code model.
    import sys
    sys.path.insert(0, str(ROOT))
    from app.mivolo import patch_timm
    patch_timm()
    patch_outlook_attention()

    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32,
    ).cpu().eval()
    install_patch_weights(model)
    export_model = MiVOLOExport(model).eval()
    sample = torch.zeros((1, 6, 384, 384), dtype=torch.float32)

    onnx_path = Path(args.onnx).resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            (sample,),
            str(onnx_path),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    print(f"[export] ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            target_platform=args.target,
        )
        # The exported graph already has fixed [1, 6, 384, 384] dimensions;
        # leave input overrides unset so Toolkit2 reads its ONNX metadata.
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="mivolo-rknn-build-") as tmpdir:
            os.chdir(tmpdir)
            try:
                ret = rknn.build(do_quantization=False)
                if ret != 0:
                    raise RuntimeError(f"RKNN build failed with code {ret}")
                ret = rknn.export_rknn(str(out_path))
                if ret != 0:
                    raise RuntimeError(f"RKNN export failed with code {ret}")
            finally:
                os.chdir(old_cwd)
        print(f"[export] RKNN saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    finally:
        rknn.release()
    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
