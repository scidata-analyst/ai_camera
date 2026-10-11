"""MiVOLO V2 age+gender (face + body dual input) — SOTA, RK3576 CPU.

Model: iitolstykh/mivolo_v2 (28.7M, 384x384, Adience gender 97.3)
Pipeline fit: YOLO person box = body, YuNet box = face.
Output per face: gender (Male/Female + conf), age (years)
"""
import os

import cv2
import numpy as np


def patch_timm():
    """mivolo lib purono timm API chay —
    timm 0.9.16 te rename/move hoiche, shim diye map koro."""
    try:
        from timm.models import _helpers as _th
        if not hasattr(_th, "remap_checkpoint") and hasattr(_th, "remap_state_dict"):
            def remap_checkpoint(model, state_dict, allow_reshape=True):
                return _th.remap_state_dict(
                    state_dict, model, allow_reshape=allow_reshape)
            _th.remap_checkpoint = remap_checkpoint
    except Exception as e:
        print(f"[mivolo] timm patch skip (_helpers): {e}")
    try:
        from timm.models import _pretrained as _tp
        from timm.models import _registry as _tr
        if not hasattr(_tp, "split_model_name_tag") and hasattr(_tr, "split_model_name_tag"):
            _tp.split_model_name_tag = _tr.split_model_name_tag
    except Exception as e:
        print(f"[mivolo] timm patch skip (_pretrained): {e}")
    try:
        # mivolo lib VOLO.__init__ ke positional args e call kore (puroto timm
        # signature); timm 0.9.16 e pos_drop_rate notun dhokay shift hoiche.
        # Keyword call e fix koro.
        import torch.nn as nn
        from timm.layers import trunc_normal_
        from timm.models.volo import VOLO
        from mivolo.model import mivolo_model as _mm

        def _fixed_init(self, layers, img_size=224, in_chans=3, num_classes=1000,
                        global_pool='token', patch_size=8, stem_hidden_dim=64,
                        embed_dims=None, num_heads=None,
                        downsamples=(True, False, False, False),
                        outlook_attention=(True, False, False, False),
                        mlp_ratio=3.0, qkv_bias=False, drop_rate=0.0,
                        attn_drop_rate=0.0, drop_path_rate=0.0,
                        norm_layer=nn.LayerNorm, post_layers=('ca', 'ca'),
                        use_aux_head=True, use_mix_token=False, pooling_scale=2):
            VOLO.__init__(
                self, layers=layers, img_size=img_size, in_chans=in_chans,
                num_classes=num_classes, global_pool=global_pool,
                patch_size=patch_size, stem_hidden_dim=stem_hidden_dim,
                embed_dims=embed_dims, num_heads=num_heads,
                downsamples=downsamples, outlook_attention=outlook_attention,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, post_layers=post_layers,
                use_aux_head=use_aux_head, use_mix_token=use_mix_token,
                pooling_scale=pooling_scale)
            im_size = img_size[0] if isinstance(img_size, tuple) else img_size
            self.patch_embed = _mm.PatchEmbed(
                img_size=im_size, stem_conv=True, stem_stride=2,
                patch_size=patch_size, in_chans=in_chans,
                hidden_dim=stem_hidden_dim, embed_dim=embed_dims[0])
            trunc_normal_(self.pos_embed, std=0.02)
            self.apply(self._init_weights)

        _mm.MiVOLOModel.__init__ = _fixed_init
    except Exception as e:
        print(f"[mivolo] init patch skip: {e}")


patch_timm()

MODEL_ID = os.getenv("MIVOLO_MODEL", "iitolstykh/mivolo_v2")


class MiVOLOAgeGender:
    def __init__(self, model_id: str = MODEL_ID):
        import time  # noqa: F401 (cache timestamp er jonno)
        import torch

        # LAG FIX: torch thread cap — default sob core nile
        # YOLO+ONNX er sathe fight kore sob slow hoy
        try:
            nth = int(os.getenv("TORCH_THREADS",
                                str(min(4, (os.cpu_count() or 4)))))
            torch.set_num_threads(max(1, nth))
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        self.enabled = os.getenv("MIVOLO_ENABLE", "1") == "1"
        # Face cache: ekoi face e bar bar heavy model chalano lage na.
        # IoU match + refresh window per refresh_sec por 1 bar re-check.
        self.refresh_sec = float(os.getenv("MIVOLO_REFRESH", "1.0"))
        self.iou_thr = float(os.getenv("MIVOLO_IOU", "0.4"))
        # LAG FIX: 1 infer call e max koyta face e model cholbe.
        # baki gulo stale/cache reuse -> 5 face e 5s block hoy na.
        self.max_faces = int(os.getenv("MIVOLO_MAX_FACES", "1"))
        # 2-sec window: oi somoyer readings er median/majority publish hoy.
        self.window_sec = float(os.getenv("MIVOLO_WINDOW", "2.0"))
        # Blur threshold: er niche sharpness hole refresh off (movement blur
        # e vul age ase). 0 korle check off.
        self.blur_thr = float(os.getenv("MIVOLO_BLUR", "80"))
        self._cache = []  # {box, gender, gconf, age, t}
        self.runs = 0   # model forward count (cache miss)
        self.hits = 0   # cache hit count
        self.last_ms = 0.0  # last forward time
        self.model = None
        self.processor = None
        self.torch = torch
        self.id2label = {0: "Female", 1: "Male"}
        if not self.enabled:
            return
        try:
            from transformers import (AutoConfig, AutoImageProcessor,
                                      AutoModelForImageClassification)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            raw = getattr(config, "gender_id2label", None) or {}
            self.id2label = {int(k): str(v).capitalize() for k, v in raw.items()} or self.id2label
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=True)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id, trust_remote_code=True,
                torch_dtype=torch.float32).to("cpu").eval()
            print(f"[mivolo] loaded {model_id} labels={self.id2label}")
        except Exception as e:
            print(f"[mivolo] disabled: {e}")
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area - inter, 1e-6)

    def _cached(self, box):
        """Fresh cache hit thakle (gender, gconf, age) dao, naile None."""
        import time
        now = time.time()
        best, best_iou = None, 0.0
        for c in self._cache:
            iou = self._iou(box, c["box"])
            if iou > best_iou:
                best, best_iou = c, iou
        if best is not None and best_iou >= self.iou_thr \
                and now - best["t"] < self.refresh_sec:
            best["box"] = box  # track movement
            return best["gender"], best["gconf"], best["age"]
        return None

    def _stale(self, box):
        """Somoy par holeo ager value (blur/side frame e reuse er jonno)."""
        best, best_iou = None, 0.0
        for c in self._cache:
            iou = self._iou(box, c["box"])
            if iou > best_iou:
                best, best_iou = c, iou
        if best is not None and best_iou >= self.iou_thr:
            best["box"] = box
            return best["gender"], best["gconf"], best["age"]
        return None

    @staticmethod
    def _blur_score(crop: np.ndarray) -> float:
        """Laplacian variance — kom mane blurry. (0-1000+ scale)"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _store(self, box, gender, gconf, age):
        """1-sec window median (age) + majority (gender).

        Window er ভিতরে jotogulo reading, tar median/majority = stable result.
        Window khali thakle ager stable value thake (Unknown hoy na).
        """
        import collections
        import statistics
        import time
        now = time.time()
        for c in self._cache:
            if self._iou(box, c["box"]) >= self.iou_thr:
                c["ages"] = [(t, a) for (t, a) in c.get("ages", []) + [(now, age)]
                             if now - t <= max(self.window_sec, 5.0)][-10:]
                c["genders"] = [(t, g) for (t, g) in c.get("genders", []) + [(now, gender)]
                                if now - t <= max(self.window_sec, 5.0)][-10:]
                win_a = [a for (t, a) in c["ages"] if now - t <= self.window_sec]
                win_g = [g for (t, g) in c["genders"] if now - t <= self.window_sec]
                med = int(statistics.median(win_a)) if win_a else c.get("age", age)
                top = collections.Counter(win_g).most_common(1)[0][0] if win_g \
                    else c.get("gender", gender)
                c.update({"box": box, "gender": top,
                          "gconf": gconf, "age": med, "t": now})
                return top, med
        self._cache.append({"box": box, "gender": gender,
                            "gconf": gconf, "age": age,
                            "ages": [(now, age)], "genders": [(now, gender)],
                            "t": now})
        self._cache = self._cache[-20:]  # memory bound
        return gender, age

    @staticmethod
    def _body_for_face(face_box, bodies):
        """Face center je human(body) box er ভিতরে, seta; na pele None."""
        x1, y1, x2, y2 = face_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for b in bodies:
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return b
        return None

    def predict_frame(self, frame: np.ndarray, faces, bodies,
                      min_side: int = 60):
        """faces e gender/age bosiye dey.

        - Fresh cache -> model skip (fast).
        - Refresh time holeo blurry/side face hole ager value reuse (vul age off).
        - Sudhu clear + frontal NOTUN frame e model chole.
        - Age = sesh 5 reading er median (flicker-free).
        (looking flag headpose theke ase — detector e headpose age chole.)
        """
        if self.model is None:
            return faces
        H, W = frame.shape[:2]
        # LAG FIX: boro face age (gender er jonno important), choto pore
        order = sorted(range(len(faces)), key=lambda i: -(
            (faces[i]["box"][2] - faces[i]["box"][0]) *
            (faces[i]["box"][3] - faces[i]["box"][1])))
        todo = []  # (face_dict, face_crop_or_None, body_crop_or_None)
        deferred = []  # max_faces er bahire -> stale reuse, block korbe na
        for i in order:
            fc = faces[i]
            x1, y1, x2, y2 = fc["box"]
            if min(x2 - x1, y2 - y1) < min_side:
                fc.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
                continue
            hit = self._cached(fc["box"])
            if hit is not None:  # cache: model chalano laglo na
                g, gc, a = hit
                self.hits += 1
                fc.update({"gender": g, "gconf": gc, "age": a})
                continue
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, W), min(y2, H)
            fcrop = frame[y1c:y2c, x1c:x2c]
            if fcrop.size == 0:
                fc.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
                continue
            # Refresh gate: blurry ba side-face hole ager value reuse koro.
            # (prothombar cache na thakle tao chalao — kichu na thakar cheye valo)
            stale = self._stale(fc["box"])
            if stale is not None:
                blurry = self.blur_thr > 0 and \
                    self._blur_score(fcrop) < self.blur_thr
                side = fc.get("looking") is False
                if blurry or side:
                    g, gc, a = stale
                    self.hits += 1
                    fc.update({"gender": g, "gconf": gc, "age": a})
                    continue
            b = self._body_for_face(fc["box"], bodies)
            bcrop = None
            if b is not None:
                bx1, by1 = max(b[0], 0), max(b[1], 0)
                bx2, by2 = min(b[2], W), min(b[3], H)
                if bx2 > bx1 and by2 > by1:
                    bcrop = frame[by1:by2, bx1:bx2]
            # LAG FIX: 1 frame e max_faces er besi model chalabe na
            if len(todo) >= max(1, self.max_faces):
                deferred.append(fc)
                continue
            todo.append((fc, fcrop, bcrop))

        # deferred face gulo stale/cache diye fill (0ms, block na)
        for fc in deferred:
            stale = self._stale(fc["box"])
            if stale is not None:
                g, gc, a = stale
                self.hits += 1
                fc.update({"gender": g, "gconf": gc, "age": a})
            else:
                fc.update({"gender": "Unknown", "gconf": 0.0, "age": 0})

        if not todo:
            return faces
        try:
            import time
            torch = self.torch
            t0 = time.time()
            f_in = self.processor(images=[t[1] for t in todo])["pixel_values"]
            b_in = self.processor(images=[t[2] for t in todo])["pixel_values"]
            f_in = f_in.to(dtype=torch.float32)
            b_in = b_in.to(dtype=torch.float32)
            # LAG FIX: inference_mode + no_grad er cheye kom overhead
            with torch.inference_mode():
                out = self.model(faces_input=f_in, body_input=b_in)
            self.last_ms = round((time.time() - t0) * 1000, 1)
            self.runs += 1
            ages = out.age_output.detach().cpu().view(-1).tolist()
            gidx = out.gender_class_idx.detach().cpu().view(-1).tolist()
            gprob = out.gender_probs.detach().cpu().view(-1).tolist()
            for (fc, _, _), a, gi, gp in zip(todo, ages, gidx, gprob):
                g = self.id2label.get(int(gi), "Unknown")
                g = "Male" if g.lower().startswith("m") else (
                    "Female" if g.lower().startswith("f") else g)
                gc = round(float(gp), 3)
                ag = max(0, min(int(round(float(a))), 100))
                top, med = self._store(fc["box"], g, gc, ag)
                fc.update({"gender": top, "gconf": gc, "age": med})
        except Exception as e:
            print(f"[mivolo] predict fail: {e}")
            for fc, _, _ in todo:
                fc.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
        return faces
