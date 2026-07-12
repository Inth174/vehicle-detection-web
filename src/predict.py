"""
==============================================================
predict.py
Image / Video Prediction
==============================================================
"""

import cv2

from src.detect import VehicleDetector
from src.tracker import MultiObjectTracker
from src.direction import DirectionDetector


class Predictor:

    def __init__(
        self,
        model_path,
        tracker_type="deepsort",
        conf=0.25,
        allowed_direction="DOWN"
    ):

        self.detector = VehicleDetector(
            model_path=model_path,
            conf=conf
        )

        self.tracker = MultiObjectTracker(
            tracker_type=tracker_type
        )

        self.direction = DirectionDetector(
            allowed_direction=allowed_direction
        )

    # =====================================================
    # IMAGE
    # =====================================================

    def predict_image(
        self,
        image
    ):

        result = self.detector.detect_image(image)

        return {
            "detections": result["detections"],
            "vehicle_count": result["vehicle_count"],
            "fps": result["fps"],
            "processing_time": result["processing_time"]
        }

    # =====================================================
    # VIDEO FRAME
    # =====================================================

    def predict_frame(
        self,
        frame
    ):

        detect_result = self.detector.detect_frame(frame)

        tracks = self.tracker.update(
            detect_result["detections"],
            frame
        )

        outputs = []

        # ====== BỔ SUNG ======
        wrong_way_count = 0
        # =====================

        for track in tracks:

            self.direction.update(
                track["track_id"],
                track["bbox"]
            )

            direction = self.direction.get_direction(
                track["track_id"]
            )

            is_wrong = self.direction.is_wrong_way(
                track["track_id"]
            )

            # ====== BỔ SUNG ======
            if is_wrong:
                wrong_way_count += 1
            # =====================

            outputs.append({
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "class_id": track["class_id"],
                "class_name": track["class_name"],
                "confidence": track["confidence"],
                "direction": direction,
                "wrong_way": is_wrong
            })

        return {
            "tracks": outputs,
            "vehicle_count": len(outputs),

            # ====== BỔ SUNG ======
            "wrong_way_count": wrong_way_count,
            # =====================

            "fps": detect_result["fps"],
            "processing_time": detect_result["processing_time"]
        }

    # =====================================================
    # VIDEO
    # =====================================================

    def predict_video(
        self,
        video_path
    ):

        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield self.predict_frame(frame)

        cap.release()
