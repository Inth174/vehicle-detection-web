"""
==============================================================
train.py
YOLOv8 Training
==============================================================
"""

from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    yaml_path,
    project_dir,
    run_name,
    model_name="yolov8s.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=2,
):

    yaml_path = Path(yaml_path)
    project_dir = Path(project_dir)

    model = YOLO(model_name)

    results = model.train(
        
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True

    )

    best_model = (
        project_dir
        / run_name
        / "weights"
        / "best.pt"
    )
    
    last_model = (
        project_dir
        / run_name
        / "weights"
        / "last.pt"
    )

    return {
        "results": results,
        "best_model": best_model,
        "last_model": last_model
    }