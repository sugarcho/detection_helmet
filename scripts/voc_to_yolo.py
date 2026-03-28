import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil

# ===== 設定 =====
dataset_dir    = Path("/home/cho/git_repo/detection_helmet/datasets")
annotations_dir = dataset_dir / "annotations"
images_dir     = dataset_dir / "images"
output_dir     = dataset_dir / "helmet_yolo"

classes = ["helmet", "head", "person"]
TRAIN_RATIO = 0.8
SEED = 42

# ===== 建立資料夾 =====
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# ===== 座標轉換（VOC → YOLO normalized） =====
def convert_bbox(img_w, img_h, xmin, xmax, ymin, ymax):
    """回傳 (cx, cy, bw, bh)，全部已除以圖片寬高，範圍 0~1"""
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh

# ===== 處理單一 xml =====
def process_xml(xml_file, split):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    img_path = images_dir / image_name

    # 圖片不存在就跳過
    if not img_path.exists():
        print(f"⚠️  Image not found, skipping: {img_path}")
        return 0

    size_node = root.find("size")
    img_w = int(size_node.find("width").text)
    img_h = int(size_node.find("height").text)

    label_lines = []

    for obj in root.iter("object"):
        cls = obj.find("name").text.strip().lower()
        if cls not in classes:
            continue

        cls_id = classes.index(cls)
        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        xmax = float(box.find("xmax").text)
        ymin = float(box.find("ymin").text)
        ymax = float(box.find("ymax").text)

        # 邊界檢查：防止標注超出圖片範圍
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            print(f"⚠️  Invalid bbox in {xml_file.name}, skipping object")
            continue

        cx, cy, bw, bh = convert_bbox(img_w, img_h, xmin, xmax, ymin, ymax)
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 沒有有效標注就跳過（不複製圖片）
    if not label_lines:
        return 0

    # 複製圖片
    shutil.copy(img_path, output_dir / "images" / split / image_name)

    # 寫 label（用 Path.stem 處理任何副檔名）
    label_path = output_dir / "labels" / split / (Path(image_name).stem + ".txt")
    label_path.write_text("\n".join(label_lines))

    return 1

# ===== 主流程 =====
xml_files = list(annotations_dir.glob("*.xml"))
random.seed(SEED)
random.shuffle(xml_files)

split_idx   = int(len(xml_files) * TRAIN_RATIO)
train_files = xml_files[:split_idx]
val_files   = xml_files[split_idx:]

print(f"Total XML: {len(xml_files)}  →  train: {len(train_files)}, val: {len(val_files)}")

train_count = sum(process_xml(f, "train") for f in train_files)
val_count   = sum(process_xml(f, "val")   for f in val_files)

print(f"✅ Done!  train images: {train_count},  val images: {val_count}")
print(f"   Skipped (no valid labels or missing image): "
      f"{len(xml_files) - train_count - val_count}")
