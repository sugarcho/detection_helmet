# inference_images.py
# Run helmet detection inference on test images using YOLOv8
# Place this file in: detection_helmet/scripts/

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent    # detection_helmet/
WEIGHTS     = BASE_DIR / "runs" / "detect" / "runs" / "train" / "helmet_v1" / "weights" / "best.pt"
TEST_IMAGES = BASE_DIR / "data" / "test" / "images"
OUTPUT_DIR  = BASE_DIR / "outputs" / "inference_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRES  = 0.25
DEVICE      = "0" if torch.cuda.is_available() else "cpu"

CLASS_COLORS = {
    0: (0, 255, 0),     # helmet  – green
    1: (0, 0, 255),     # head    – red
    2: (255, 165, 0),   # person  – orange
}
CLASS_NAMES = ["helmet", "head", "person"]


# ── Draw boxes & summary ──────────────────────────────────────────────────
def annotate_frame(img, result):
    annotated = img.copy()
    counts = {i: 0 for i in range(len(CLASS_NAMES))}

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            counts[cls_id] += 1

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Detection count summary (top-right)
    h, w = annotated.shape[:2]
    for i, (cls_id, cnt) in enumerate(counts.items()):
        text = f"{CLASS_NAMES[cls_id]}: {cnt}"
        cv2.putText(annotated, text, (w - 175, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated, counts


# ── Main ──────────────────────────────────────────────────────────────────
def run_inference():
    if not WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS}")
        print("        Make sure training is complete and best.pt exists.")
        return
    if not TEST_IMAGES.exists():
        print(f"[ERROR] Test images folder not found: {TEST_IMAGES}")
        return

    model = YOLO(str(WEIGHTS))
    print(f"[INFO] Model loaded: {WEIGHTS}")
    print(f"[INFO] Device: {DEVICE}\n")

    image_paths = sorted([
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    if not image_paths:
        print("[ERROR] No images found in:", TEST_IMAGES)
        return

    print(f"[INFO] Found {len(image_paths)} test image(s). Running inference ...\n")

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[SKIP] Cannot read: {img_path.name}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False
        )

        annotated, counts = annotate_frame(img_bgr, results[0])

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  {img_path.name:40s} | helmet={counts[0]}  head={counts[1]}  person={counts[2]}")

    print(f"\n[DONE] Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Helmet Detection – Image Inference (YOLOv8)")
    print("=" * 55)
    run_inference()