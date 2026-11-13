"""Live face + emotion recognition from a webcam (or a video file) with OpenCV.

Usage:
    python webcam.py                 # default camera, 320 px detector input
    python webcam.py --source 1      # another camera
    python webcam.py --source clip.mp4 --size 640
Press q to quit.
"""
import argparse
import time
import urllib.request  # hsemotion_onnx needs urllib.request imported to download its model
from pathlib import Path

import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from ultralytics import YOLO

ROOT = Path(__file__).parent
FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"


def load_models():
    weights = ROOT / "data" / "yolov8n-face.pt"
    if not weights.exists():
        weights.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FACE_URL, weights)
    return YOLO(str(weights)), HSEmotionRecognizer(model_name="enet_b0_8_best_afew")


def analyze(frame, face_model, emo, size=320):
    """Detect faces, classify each face's emotion and draw the result on the frame."""
    found = []
    r = face_model(frame, imgsz=size, verbose=False)[0]
    for x1, y1, x2, y2 in r.boxes.xyxy.int().tolist():
        if x2 <= x1 or y2 <= y1:
            continue
        rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        name, scores = emo.predict_emotions(rgb, logits=False)
        found.append((name, float(scores.max())))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {scores.max():.0%}", (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--size", type=int, default=320, help="detector input size (320 is ~3x faster than 640)")
    args = ap.parse_args()

    face_model, emo = load_models()
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source!r}")
    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _ = analyze(frame, face_model, emo, args.size)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - last) if fps else 1 / (now - last)
        last = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face emotion detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
