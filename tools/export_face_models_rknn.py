from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAIRFACE_URL = (
    "https://huggingface.co/livinNector/fairface/resolve/main/fairface.onnx")
DLIB_AGE_URL = (
    "https://raw.githubusercontent.com/mowshon/age-and-gender/master/"
    "src/age_and_gender/models/age-v1.onnx")
MIVOLO_ZIP_URL = (
    "https://github.com/yanghaojie0817/Age_Gender-Prediction-by-Mivolo/"
    "releases/download/V1.0.0/model_onnx.zip")
MIVOLO_ZIP_SHA256 = (
    "b253569f7f3e47ddf489ac36b4d14af9c5d6040582c92144b8e69ac93e637dd0")
MIVOLO_ONNX_SHA256 = (
    "209c960a71ed51930763984c27912d051e58810170b46b249a3a5235c069dd0f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_mivolo() -> Path:
    onnx = ROOT / "models/mivolo-face.onnx"
    report = ROOT / "models/mivolo-face.json"
    if onnx.is_file() and report.is_file():

        if sha256(onnx) != MIVOLO_ONNX_SHA256:

            raise SystemExit(f"Checksum mismatch: {onnx}")
        return onnx
    with tempfile.TemporaryDirectory(prefix="mivolo-onnx-") as tmp:
        archive = Path(tmp) / "model_onnx.zip"
        print(f"[download] {MIVOLO_ZIP_URL}")
        urllib.request.urlretrieve(MIVOLO_ZIP_URL, archive)
        if sha256(archive) != MIVOLO_ZIP_SHA256:

            raise SystemExit(f"Checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract("model_onnx/mivolo_face_only_rknn.onnx", tmp)
            bundle.extract("model_onnx/mivolo_face_only_rknn.json", tmp)
        source = Path(tmp) / "model_onnx/mivolo_face_only_rknn.onnx"
        if sha256(source) != MIVOLO_ONNX_SHA256:

            raise SystemExit(f"Checksum mismatch: {source}")
        source.replace(onnx)
        (Path(tmp) / "model_onnx/mivolo_face_only_rknn.json").replace(report)
    return onnx


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

    age = ROOT / "models/dlib-age-v1.onnx"
    if not age.is_file():
        print(f"[download] {DLIB_AGE_URL}")
        urllib.request.urlretrieve(DLIB_AGE_URL, age)

    convert(
        age,
        ROOT / "models/dlib-age-rk3576.rknn",
        args.target, [[0, 0, 0]], [[1, 1, 1]],
        input_name="images", input_size=(1, 3, 64, 64),
    )

    mivolo = ensure_mivolo()
    metadata = json.loads(
        (ROOT / "models/mivolo-face.json").read_text(encoding="utf-8"))
    mean = [[round(value * 255, 6) for value in metadata["mean"]]]
    std = [[round(value * 255, 6) for value in metadata["std"]]]
    convert(
        mivolo,
        ROOT / "models/mivolo-face-rk3576.rknn",
        args.target, mean, std,
    )


if __name__ == "__main__":
    main()
