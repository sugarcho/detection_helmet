# prepare_test_images.py
# Copy a sample of val images into data/test/images/ for inference testing
# Place this file in: detection_helmet/scripts/

import shutil
import random
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
VAL_DIR    = BASE_DIR / "datasets" / "helmet_yolo" / "images" / "val"
TEST_DIR   = BASE_DIR / "data" / "test" / "images"
TEST_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 10   # number of images to copy

def main():
    images = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))

    if not images:
        print(f"[ERROR] No images found in: {VAL_DIR}")
        return

    sample = random.sample(images, min(NUM_IMAGES, len(images)))

    for img in sample:
        shutil.copy(img, TEST_DIR / img.name)

    print(f"[DONE] Copied {len(sample)} images to: {TEST_DIR}")

if __name__ == "__main__":
    main()