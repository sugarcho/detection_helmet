# inference_video.py
# Run helmet detection on a video file using YOLOv8
# Place this file in: detection_helmet/scripts/

import cv2
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
WEIGHTS    = BASE_DIR / "runs" / "detect" / "runs" / "train" / "helmet_v1" / "weights" / "best.pt"
VIDEO_IN   = BASE_DIR / "data" / "test_video.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "inference_video"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT  = OUTPUT_DIR / "helmet_detection_output.mp4"

CONF_THRES = 0.25
DEVICE     = "0" if torch.cuda.is_available() else "cpu"
SHOW_LIVE  = False

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 165, 0),
}
CLASS_NAMES = ["helmet", "head", "person"]


# ── Helpers ───────────────────────────────────────────────────────────────
def annotate_frame(frame, result):
    annotated = frame.copy()
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

    return annotated, counts


def draw_hud(frame, counts, frame_no, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, "Helmet Safety Detection System", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x = 10
    for cls_id, name in enumerate(CLASS_NAMES):
        cv2.putText(frame, f"{name}: {counts[cls_id]}", (x, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLASS_COLORS[cls_id], 2)
        x += 200

    if counts[1] > 0:
        cv2.putText(frame, f"WARNING: {counts[1]} worker(s) WITHOUT helmet!", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Frame {frame_no} | {fps:.1f} FPS", (w - 220, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


# Main
def run_video_inference():
    if not WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS}")
        return
    if not VIDEO_IN.exists():
        print(f"[ERROR] Video not found: {VIDEO_IN}")
        print(f"        Place your test video at: {VIDEO_IN}")
        return

    model = YOLO(str(WEIGHTS))
    print(f"[INFO] Model loaded | Device: {DEVICE}")

    cap     = cv2.VideoCapture(str(VIDEO_IN))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer  = cv2.VideoWriter(str(VIDEO_OUT),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               src_fps, (src_w, src_h))

    print(f"[INFO] {src_w}x{src_h}  {src_fps:.1f} FPS  {total_f} frames")
    print(f"[INFO] Output: {VIDEO_OUT}\n")

    frame_no = 0
    warn_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(source=frame, conf=CONF_THRES,
                                device=DEVICE, verbose=False)
        fps_now = 1.0 / max(time.perf_counter() - t0, 1e-6)

        annotated, counts = annotate_frame(frame, results[0])
        annotated = draw_hud(annotated, counts, frame_no, fps_now)
        writer.write(annotated)

        if counts[1] > 0:
            warn_frames += 1

        frame_no += 1
        if frame_no % 30 == 0:
            print(f"  {frame_no}/{total_f} ({frame_no/max(total_f,1)*100:.1f}%) "
                  f"fps={fps_now:.1f}  no_helmet={counts[1]}")

        if SHOW_LIVE:
            cv2.imshow("Helmet Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] {frame_no} frames processed | Warning frames: {warn_frames}")
    print(f"[DONE] Output saved to: {VIDEO_OUT}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Helmet Detection – Video Inference (YOLOv8)")
    print("=" * 55)
    run_video_inference()