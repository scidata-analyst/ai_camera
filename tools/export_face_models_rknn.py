from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAIRFACE_URL = (
    "https://huggingface.co/livinNector/fairface/resolve/main/fairface.onnx")


def convert(model: Path, output: Path, target: str,
            mean_values, std_values, input_name: str | None = None,
            input_size: tuple[int, ...] | None = None) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            target_platform=target,
            mean_values=mean_values,
            std_values=std_values,
        )
        if input_name and input_size:
            ret = rknn.load_onnx(
                model=str(model), inputs=[input_name],
                input_size_list=[list(input_size)],
            )
        else:
            ret = rknn.load_onnx(model=str(model))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed for {model}: {ret}")
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="face-rknn-build-") as tmp:
            os.chdir(tmp)
            try:
                ret = rknn.build(do_quantization=False)
                if ret != 0:
                    raise RuntimeError(f"build failed for {model}: {ret}")
                ret = rknn.export_rknn(str(output))
                if ret != 0:
                    raise RuntimeError(f"export_rknn failed for {output}: {ret}")
            finally:
                os.chdir(old_cwd)
        print(f"[export] saved {output} ({output.stat().st_size / 1e6:.2f} MB)")
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="rk3576")
    args = parser.parse_args()

    retinaface = ROOT / "models/RetinaFace_mobile320.onnx"
    if not retinaface.is_file():
        raise SystemExit("Missing models/RetinaFace_mobile320.onnx. Download it first.")

    fairface = ROOT / "models/fairface.onnx"
    if not fairface.is_file():
        print(f"[download] {FAIRFACE_URL}")
        urllib.request.urlretrieve(FAIRFACE_URL, fairface)

    convert(
        retinaface,
        ROOT / "models/RetinaFace_mobile320-rk3576.rknn",
        args.target, [[104, 117, 123]], [[1, 1, 1]],
    )

    convert(
        fairface,
        ROOT / "models/fairface-rk3576.rknn",
        args.target, [[0, 0, 0]], [[1, 1, 1]],
        input_name="input", input_size=(1, 3, 224, 224),
    )


if __name__ == "__main__":
    main()
