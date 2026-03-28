from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # s = small，速度與精度平衡

results = model.train(
    data="/home/cho/git_repo/detection_helmet/datasets/helmet_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    patience=10,       # 10 epoch 沒進步就 early stop
    save=True,
    project="runs/train",
    name="helmet_v1",
    exist_ok=True,
)

print("Best mAP50:", results.results_dict["metrics/mAP50(B)"])
