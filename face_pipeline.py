"""NumPy + ONNX Runtime face/emotion pipeline that mirrors web/app.js step by step.

The browser demo cannot use ultralytics or hsemotion_onnx, so every pre/post-processing
step is reimplemented by hand in JavaScript. This module is the exact same algorithm in
NumPy; train.py runs it next to the reference pipeline (ultralytics + hsemotion_onnx)
to prove both give the same faces and emotions.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet stats used by hsemotion_onnx
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMO_SIZE = 224                                               # enet_b0 input size in hsemotion_onnx
PAD_VALUE = 114                                              # ultralytics letterbox grey


def resize_bilinear(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """cv2.INTER_LINEAR semantics (half-pixel centres, edge clamp), rounded back to uint8."""
    h, w = img.shape[:2]

    def coords(n_out, n_in):
        f = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        i0 = np.floor(f).astype(np.int64)
        frac = f - i0
        frac[i0 < 0] = 0.0
        i0 = np.clip(i0, 0, n_in - 1)
        frac[i0 >= n_in - 1] = 0.0
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, frac.astype(np.float32)

    x0, x1, fx = coords(out_w, w)
    y0, y1, fy = coords(out_h, h)
    src = img.astype(np.float32)
    top = src[y0][:, x0] * (1 - fx)[None, :, None] + src[y0][:, x1] * fx[None, :, None]
    bot = src[y1][:, x0] * (1 - fx)[None, :, None] + src[y1][:, x1] * fx[None, :, None]
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def letterbox(rgb: np.ndarray, size: int):
    """Resize keeping aspect ratio and pad to size x size (same rules as ultralytics LetterBox)."""
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    resized = rgb if (new_w, new_h) == (w, h) else resize_bilinear(rgb, new_w, new_h)
    canvas[top:top + new_h, left:left + new_w] = resized
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, (new_w / w, new_h / h, left, top)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between one box a (4,) and many boxes b (N, 4), xyxy."""
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        if order:
            rest = np.array(order)
            order = list(rest[iou(boxes[i], boxes[rest]) <= iou_thr])
    return keep


def decode(out: np.ndarray, meta, img_w: int, img_h: int, conf: float = 0.25, iou_thr: float = 0.7):
    """YOLOv8 head output (1, 4+1+15, N) -> xyxy boxes in original image pixels + scores."""
    p = out[0]
    scores = p[4]
    m = scores > conf
    cx, cy, bw, bh = p[0, m], p[1, m], p[2, m], p[3, m]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    scores = scores[m]
    keep = nms(boxes, scores, iou_thr)[:300]
    boxes, scores = boxes[keep], scores[keep]
    gx, gy, left, top = meta
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gx, 0, img_w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gy, 0, img_h)
    return boxes, scores


def emotion_input(rgb: np.ndarray, box) -> np.ndarray:
    """Crop (integer box, like `xyxy.int()`), resize to 224, scale to [0, 1], ImageNet-normalise."""
    x1, y1, x2, y2 = (int(v) for v in box)
    face = resize_bilinear(rgb[y1:y2, x1:x2], EMO_SIZE, EMO_SIZE).astype(np.float32) / 255.0
    return ((face - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class BrowserPipeline:
    """Same steps as web/app.js, run with onnxruntime on the exported web models."""

    def __init__(self, detector_path, emotion_path, size: int):
        self.det = ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
        self.emo = ort.InferenceSession(str(emotion_path), providers=["CPUExecutionProvider"])
        self.size = size

    def __call__(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        x, meta = letterbox(rgb, self.size)
        out = self.det.run(None, {"images": x})[0]
        boxes, scores = decode(out, meta, w, h)
        faces = []
        good = [(b, s) for b, s in zip(boxes, scores) if int(b[2]) > int(b[0]) and int(b[3]) > int(b[1])]
        if good:
            batch = np.stack([emotion_input(rgb, b) for b, _ in good])
            probs = softmax(self.emo.run(None, {"input": batch})[0])
            for (b, s), p in zip(good, probs):
                faces.append({"box": b.tolist(), "score": float(s),
                              "emotion": EMOTIONS[int(p.argmax())], "probs": p.tolist()})
        return faces
