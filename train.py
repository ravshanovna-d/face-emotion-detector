"""Prepare the face + emotion models for the browser demo and verify them.

There is no training here: both networks are pretrained. This script
  1. downloads the models and sample images (cached in data/),
  2. exports the YOLOv8n-face detector to ONNX and stores the weights of both
     ONNX models as float16 (halves the download for the web page),
  3. checks that the browser pipeline (face_pipeline.py == web/app.js logic, run with
     onnxruntime) finds the same faces and emotions as ultralytics + hsemotion_onnx,
  4. benchmarks 640 vs 320 input size,
  5. writes web/models/*.onnx, web/samples/*, web/assets/*, web/meta.json.

Usage:  python train.py
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from face_pipeline import EMOTIONS, BrowserPipeline, iou

ROOT = Path(__file__).parent
DATA, WEB = ROOT / "data", ROOT / "web"
MODELS, SAMPLES, ASSETS = WEB / "models", WEB / "samples", WEB / "assets"
WEB_SIZE = 320                       # detector input size used in the browser
FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
EMO_NAME = "enet_b0_8_best_afew"
EMO_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
           f"models/affectnet_emotions/onnx/{EMO_NAME}.onnx")
UL_ASSETS = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/"
SAMPLE_INFO = {
    "zidane.jpg": "Zinedine Zidane & Carlo Ancelotti (Ultralytics sample image)",
    "bus.jpg": "Street scene (Ultralytics sample image)",
    "astronaut.jpg": "Astronaut Eileen Collins (NASA, public domain, via scikit-image)",
}


def download(url: str, path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading", url)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def emotion_model_path() -> Path:
    """Use the file hsemotion_onnx caches in ~/.hsemotion, downloading it if missing."""
    cached = Path(os.path.expanduser("~")) / ".hsemotion" / f"{EMO_NAME}.onnx"
    try:
        onnx.checker.check_model(str(cached))
    except Exception:
        cached.unlink(missing_ok=True)
        download(EMO_URL, cached)
    return cached


def fp16_weights(src: Path, dst: Path, min_size: int = 1024) -> None:
    """Store large float32 initializers as float16 and cast them back to float32 at load time.
    All maths still runs in float32, so any runtime that supports Cast can run the model."""
    model = onnx.load(str(src))
    g = model.graph
    casts, new_inits = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= min_size:
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16"))
            casts.append(helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def prepare_samples() -> dict[str, np.ndarray]:
    from skimage import data as skdata
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name in ("zidane.jpg", "bus.jpg"):
        download(UL_ASSETS + name, DATA / name)
        shutil.copy(DATA / name, SAMPLES / name)
    astro = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(SAMPLES / "astronaut.jpg"), astro, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # Read back the exact files the web page ships with.
    return {n: cv2.imread(str(SAMPLES / n)) for n in SAMPLE_INFO}


class Reference:
    """The original notebook pipeline: ultralytics face detector + hsemotion_onnx."""

    def __init__(self, detector):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        from ultralytics import YOLO
        self.det = YOLO(str(detector), task="pose")
        self.emo = HSEmotionRecognizer(model_name=EMO_NAME)

    def __call__(self, bgr: np.ndarray, size: int):
        r = self.det(bgr, imgsz=size, verbose=False)[0]
        faces = []
        for (x1, y1, x2, y2), xyxy, conf in zip(r.boxes.xyxy.int().tolist(), r.boxes.xyxy.tolist(),
                                                r.boxes.conf.tolist()):
            if x2 <= x1 or y2 <= y1:
                continue
            rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            name, probs = self.emo.predict_emotions(rgb, logits=False)
            faces.append({"box": xyxy, "score": conf, "emotion": name, "probs": probs.tolist()})
        return faces


def compare(ref: list[dict], ours: list[dict]) -> dict:
    """Match faces by IoU and compare boxes, labels and probabilities."""
    res = {"ref_faces": len(ref), "ours_faces": len(ours), "matched": 0,
           "min_iou": 1.0, "max_box_px": 0.0, "same_label": 0, "max_prob_diff": 0.0}
    if not ref or not ours:
        return res
    ob = np.array([f["box"] for f in ours])
    for f in ref:
        ious = iou(np.array(f["box"]), ob)
        j = int(ious.argmax())
        if ious[j] < 0.5:
            continue
        g = ours[j]
        res["matched"] += 1
        res["min_iou"] = min(res["min_iou"], float(ious[j]))
        res["max_box_px"] = max(res["max_box_px"], float(np.abs(np.array(f["box"]) - g["box"]).max()))
        res["same_label"] += int(f["emotion"] == g["emotion"])
        res["max_prob_diff"] = max(res["max_prob_diff"],
                                   float(np.abs(np.array(f["probs"]) - g["probs"]).max()))
    return res


def draw(bgr: np.ndarray, faces: list[dict]) -> np.ndarray:
    out = bgr.copy()
    t = max(2, round(max(out.shape[:2]) / 400))
    for f in faces:
        x1, y1, x2, y2 = (int(v) for v in f["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (229, 70, 79), t)
        label = f"{f['emotion']} {max(f['probs']):.0%}"
        scale = t * 0.35
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        y = max(y1, th + 8)
        cv2.rectangle(out, (x1, y - th - 8), (x1 + tw + 8, y), (229, 70, 79), -1)
        cv2.putText(out, label, (x1 + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t)
    return out


def timed(fn, n: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times)) * 1000   # median is robust to background load


def main():
    from ultralytics import YOLO
    for d in (DATA, MODELS, ASSETS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Models ------------------------------------------------------------------
    pt = download(FACE_URL, DATA / "yolov8n-face.pt")
    det_onnx = {}
    for size in (WEB_SIZE, 640):
        path = DATA / f"yolov8n-face-{size}.onnx"
        if not path.exists():
            exported = Path(YOLO(str(pt)).export(format="onnx", imgsz=size, opset=17, simplify=True))
            exported.replace(path)
        det_onnx[size] = path
    emo_src = emotion_model_path()
    fp16_weights(det_onnx[WEB_SIZE], MODELS / "face_detector.onnx")
    fp16_weights(emo_src, MODELS / "emotion.onnx")
    sizes_mb = {p.name: round(p.stat().st_size / 1e6, 1) for p in MODELS.glob("*.onnx")}
    print("Web models (MB):", sizes_mb)

    # 2. Verification -----------------------------------------------------------
    images = prepare_samples()
    ref_pt = Reference(pt)
    ref_onnx = Reference(det_onnx[WEB_SIZE])          # ultralytics running the fp32 ONNX export
    browser = BrowserPipeline(MODELS / "face_detector.onnx", MODELS / "emotion.onnx", WEB_SIZE)
    verification, results = [], {}
    for name, bgr in images.items():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ours = browser(rgb)
        results[name] = ours
        for ref_name, ref in (("ultralytics .pt", ref_pt), ("ultralytics .onnx", ref_onnx)):
            c = compare(ref(bgr, WEB_SIZE), ours)
            verification.append({"image": name, "reference": ref_name, **c})
        cv2.imwrite(str(ASSETS / f"result_{Path(name).stem}.jpg"), draw(bgr, ours),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name, [(f["emotion"], round(max(f["probs"]), 2)) for f in ours])
    print("\nVerification (browser pipeline vs reference):")
    for v in verification:
        print(f"  {v['image']:14s} vs {v['reference']:18s} faces {v['ref_faces']}/{v['ours_faces']} "
              f"matched {v['matched']}  min IoU {v['min_iou']:.3f}  max box diff {v['max_box_px']:.1f}px  "
              f"same label {v['same_label']}/{v['matched']}  max prob diff {v['max_prob_diff']:.3f}")

    # 3. Speed: 640 vs 320 ------------------------------------------------------
    del ref_onnx, browser
    gc.collect()
    frame = images["zidane.jpg"]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bench = []
    for size in (640, 320):
        faces = ref_pt(frame, size)
        bench.append({"pipeline": "ultralytics (PyTorch) + HSEmotion", "size": size,
                      "ms": round(timed(lambda: ref_pt(frame, size)), 1), "faces": len(faces)})
    for size in (640, 320):
        p = BrowserPipeline(det_onnx[size], emo_src, size)
        bench.append({"pipeline": "ONNX Runtime (browser logic)", "size": size,
                      "ms": round(timed(lambda: p(rgb)), 1), "faces": len(p(rgb))})
        del p
        gc.collect()
    print("\nSpeed on zidane.jpg (CPU, median of 10 runs):")
    for b in bench:
        print(f"  {b['pipeline']:36s} {b['size']}px  {b['ms']:7.1f} ms/frame (median)  {1000 / b['ms']:5.1f} FPS  faces={b['faces']}")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{'PyTorch' if 'PyTorch' in b['pipeline'] else 'ONNX'} @ {b['size']}" for b in bench]
    ax.barh(labels, [b["ms"] for b in bench], color=["#a5a1ff", "#4f46e5", "#a5a1ff", "#4f46e5"])
    for i, b in enumerate(bench):
        ax.text(b["ms"], i, f" {b['ms']:.0f} ms", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ms per frame (detection + emotion, CPU)")
    ax.set_title("Input size 640 vs 320")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / "speed.png", dpi=130)
    plt.close()

    # 4. Meta for the web page ---------------------------------------------------
    total = {k: sum(v[k] for v in verification) for k in ("ref_faces", "matched", "same_label")}
    meta = {
        "title": "Face Emotion Detector",
        "subtitle": "Finds faces in a photo or live webcam video and recognises 8 facial expressions. "
                    "Both neural networks run in your browser with ONNX Runtime Web.",
        "tags": ["Computer vision", "YOLOv8-face", "EfficientNet-B0", "ONNX Runtime Web", "Webcam"],
        "input_size": WEB_SIZE,
        "conf": 0.25, "iou": 0.7,
        "emotions": EMOTIONS,
        "samples": [{"file": f"samples/{n}", "label": Path(n).stem.capitalize(), "credit": c}
                    for n, c in SAMPLE_INFO.items()],
        "model_sizes_mb": sizes_mb,
        "verification": verification,
        "verification_total": total,
        "benchmark": bench,
        "sample_results": {n: [{"emotion": f["emotion"], "confidence": round(max(f["probs"]), 3)}
                               for f in r] for n, r in results.items()},
        "steps": [
            "Detected objects with YOLO11n, then switched to a face-specific YOLOv8n model.",
            "Cropped each face and classified its expression with HSEmotion (EfficientNet-B0, 8 emotions, AffectNet + AFEW).",
            "Wrapped both models into one analyze(frame) pipeline and ran it on photos and a webcam stream.",
            "Benchmarked 640 vs 320 input size: 320 is about 2x faster on CPU and still finds both faces in the test image.",
            "Exported the detector to ONNX and stored both models' weights as float16 to halve the download.",
            "Re-implemented letterboxing, YOLO decoding + NMS and the HSEmotion preprocessing in JavaScript, "
            "and verified the same logic in NumPy against ultralytics + hsemotion_onnx.",
        ],
        "repo_url": "https://github.com/ravshanovna-d/face-emotion-detector",
        "notebook_url": "https://github.com/ravshanovna-d/face-emotion-detector/blob/main/notebook.ipynb",
        "footer": 'Built by <a href="https://github.com/ravshanovna-d">ravshanovna-d</a> · '
                  "Models: YOLOv8n-face (AGPL-3.0) · HSEmotion (Apache-2.0)",
    }
    (WEB / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    web_mb = sum(f.stat().st_size for f in WEB.rglob("*") if f.is_file()) / 1e6
    print(f"\nwrote web/ ({web_mb:.1f} MB); faces matched {total['matched']}/{total['ref_faces']}, "
          f"same emotion {total['same_label']}/{total['matched']}")


if __name__ == "__main__":
    main()
