"""yolo11n.pt -> RK3576 .rknn (NPU model) export.

PC (x86) te chalao — rknn-toolkit2 conversion x86 te supported,
cholbe sudhu board e (RKNNLite + /dev/rknpu).
Output: models/yolo11n-rk3576.rknn (compose er RKNN_MODEL er sathe mile)

Usage:
  pip install rknn-toolkit2 ultralytics onnx
  python tools/export_rknn.py                          # fp (default, safe)
  python tools/export_rknn.py --quant i8 --dataset ./calib.txt --imgsz 320
  python tools/export_rknn.py --model yolo11s.pt --imgsz 480

Note (app/app/detector.py _run_rknn er sathe contract):
  - input : NCHW RGB float32 0-1, square imgsz (letterbox app code e hoy)
  - output: (1,84,N) raw logits (sigmoid app code e hoy) — COCO 80 class
"""
import argparse
import os
import shutil


def export_onnx(pt: str, imgsz: int, workdir: str) -> str:
    from ultralytics import YOLO
    print(f"[export] pt -> onnx: {pt} imgsz={imgsz}")
    # dynamic=False: static (1,3,H,W) — RKNN convert stable hoy
    out = YOLO(pt).export(format="onnx", imgsz=imgsz, opset=12,
                          simplify=True, dynamic=False)
    dst = os.path.join(workdir, "yolo11n.onnx")
    if os.path.abspath(out) != os.path.abspath(dst):
        shutil.move(out, dst)
    # ultralytics .cache dir poriskar (repo nogra hoy na)
    for p in (os.path.splitext(pt)[0] + ".cache",):
        if os.path.exists(p):
            os.remove(p)
    return dst


def build_rknn(onnx_path: str, out_path: str, imgsz: int,
               target: str = "rk3576", quant: str = "fp",
               dataset: str | None = None) -> str:
    from rknn.api import RKNN
    do_quant = quant.lower() in ("i8", "int8")
    if do_quant and not dataset:
        raise SystemExit("[export] --quant i8 hole --dataset calib.txt lagbe "
                         "(proti line e 1ta jpg path)")
    print(f"[export] onnx -> rknn: target={target} quant={'i8' if do_quant else 'fp'}")
    rknn = RKNN(verbose=True)
    # mean/std: app code already /255 float32 pathay, tai norm off (std=1)
    rknn.config(target_platform=target,
                mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])
    ret = rknn.load_onnx(model=onnx_path, inputs=["images"],
                         input_size_list=[[3, imgsz, imgsz]])
    if ret != 0:
        raise SystemExit(f"[export] load_onnx fail ret={ret}")
    ret = rknn.build(do_quantization=do_quant,
                     dataset=dataset if do_quant else None)
    if ret != 0:
        raise SystemExit(f"[export] build fail ret={ret}")
    ret = rknn.export_rknn(out_path)
    rknn.release()
    if ret != 0:
        raise SystemExit(f"[export] export_rknn fail ret={ret}")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[export] OK: {out_path} ({size_mb:.1f}MB)")
    print("[export] board e test: RKNN_MODEL=" + out_path +
          " YOLO_BACKEND=rknn docker compose up --build")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="YOLO pt -> RK3576 rknn")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--target", default="rk3576")
    ap.add_argument("--quant", default="fp", choices=["fp", "i8", "int8"])
    ap.add_argument("--dataset", default=None,
                    help="int8 calibration list (txt, proti line jpg)")
    ap.add_argument("--out", default="models/yolo11n-rk3576.rknn")
    ap.add_argument("--workdir", default="models/.export_tmp")
    ap.add_argument("--keep-onnx", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    onnx_path = export_onnx(a.model, a.imgsz, a.workdir)
    try:
        build_rknn(onnx_path, a.out, a.imgsz, a.target, a.quant, a.dataset)
    finally:
        if not a.keep_onnx and os.path.exists(a.workdir):
            shutil.rmtree(a.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
