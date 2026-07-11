"""
==============================================================
detect.py
YOLOv8 Vehicle Detection
==============================================================
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import time


class VehicleDetector:

    def __init__(self, model_path, conf=0.25, device=None):

        self.model_path = Path(model_path)
        self.conf = conf
        self.model = YOLO(str(self.model_path))

        if device is not None:
            self.model.to(device)

    # ----------------------------------------------------------
    # Image
    # ----------------------------------------------------------

    def detect_image(self, image):

        start = time.time()

        results = self.model.predict(
            source=image,
            conf=self.conf,
            verbose=False
        )

        process_time = time.time() - start
        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls = int(box.cls.item())
                score = float(box.conf.item())
                class_name = self.model.names[cls]

                xmin, ymin, xmax, ymax = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                detections.append({
                    "bbox": [
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax)
                    ],

                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": score
                })

        return {

            "detections": detections,
            "vehicle_count": len(detections),
            "processing_time": process_time,

            "fps": (
                1 / process_time
                if process_time > 0 else 0
            )

        }

    # ----------------------------------------------------------
    # Video Frame
    # ----------------------------------------------------------

    def detect_frame(self, frame):
        return self.detect_image(frame)