from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX -> RKNN")
    parser.add_argument("--model", required=True, help="Fixed-shape ONNX input")
    parser.add_argument("--out", help="RKNN output (defaults beside ONNX)")
    parser.add_argument("--target", default="rk3576")
    args = parser.parse_args()

    model = Path(args.model).resolve()
    out = Path(args.out).resolve() if args.out else model.with_name(
        f"{model.stem}-rk3576.rknn")
    if not model.is_file():

        raise SystemExit(f"ONNX model not found: {model}")
    out.parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(target_platform=args.target)
        ret = rknn.load_onnx(model=str(model))
        if ret != 0:

            raise RuntimeError(f"load_onnx failed: {ret}")
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="onnx-rknn-build-") as tmp:
            os.chdir(tmp)
            try:
                ret = rknn.build(do_quantization=False)
                if ret != 0:

                    raise RuntimeError(f"build failed: {ret}")
                ret = rknn.export_rknn(str(out))
                if ret != 0:

                    raise RuntimeError(f"export_rknn failed: {ret}")
            finally:
                os.chdir(old_cwd)
        print(f"[export] RKNN saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
