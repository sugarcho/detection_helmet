# make_test_video.py
# Create a test video by combining val images into an mp4
# Place this file in: detection_helmet/scripts/

import cv2
import random
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
VAL_DIR    = BASE_DIR / "datasets" / "helmet_yolo" / "images" / "val"
VIDEO_OUT  = BASE_DIR / "data" / "test_video.mp4"
VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)

NUM_IMAGES  = 50    # how many val images to include
FPS         = 5     # frames per second (low = easier to see each detection)
IMG_SIZE    = (640, 640)

def main():
    images = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))

    if not images:
        print(f"[ERROR] No images found in: {VAL_DIR}")
        return

    sample = random.sample(images, min(NUM_IMAGES, len(images)))
    sample.sort()  # consistent order

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, FPS, IMG_SIZE)

    for i, img_path in enumerate(sample):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.resize(frame, IMG_SIZE)
        writer.write(frame)
        print(f"  [{i+1}/{len(sample)}] Added: {img_path.name}")

    writer.release()
    print(f"\n[DONE] Video saved to: {VIDEO_OUT}")
    print(f"[INFO] {len(sample)} frames  |  {FPS} FPS  |  ~{len(sample)//FPS}s duration")

if __name__ == "__main__":
    main()