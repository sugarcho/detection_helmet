from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="D:/Github/Cho_repo/detection_helmet/datasets/helmet_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.01,
        patience=10,
        save=True,
        project="runs/train",
        name="helmet_v1",
        exist_ok=True,
        device="0",
        workers=4,      # Windows 建議設 4
    )

    print("Best mAP50:", results.results_dict["metrics/mAP50(B)"])