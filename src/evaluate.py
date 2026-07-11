"""
==============================================================
evaluate.py
System Evaluation
==============================================================
"""

from pathlib import Path
import csv
from statistics import mean

from src.predict import Predictor


class Evaluator:

    def __init__(
        self,
        model_path,
        tracker_type="deepsort",
        conf=0.25,
        allowed_direction="DOWN"
    ):

        self.predictor = Predictor(
            model_path=model_path,
            tracker_type=tracker_type,
            conf=conf,
            allowed_direction=allowed_direction
        )

    # ======================================================
    # IMAGE
    # ======================================================

    def evaluate_image(
        self,
        image
    ):

        result = self.predictor.predict_image(image)

        return {
            "vehicle_count": result["vehicle_count"],
            "fps": result["fps"],
            "processing_time": result["processing_time"]
        }

    # ======================================================
    # VIDEO
    # ======================================================

    def evaluate_video(
        self,
        video_path
    ):

        fps_list = []
        time_list = []
        current_vehicle = []
        wrong_way = []
        total_frames = 0

        for result in self.predictor.predict_video(video_path):

            total_frames += 1
            fps_list.append(result["fps"])
            time_list.append(result["processing_time"])

            current_vehicle.append(
                result["vehicle_count"]
            )

            wrong_way.append(
                sum(
                    t["wrong_way"]
                    for t in result["tracks"]
                )
            )

        return {
            "total_frames": total_frames,
            "average_fps": mean(fps_list)

            if fps_list else 0,
            "average_processing_time": mean(time_list)

            if time_list else 0,
            "average_vehicle_count": mean(current_vehicle)

            if current_vehicle else 0,
            "total_wrong_way": sum(wrong_way)
        }

    # ======================================================
    # SAVE CSV
    # ======================================================

    def save_csv(
        self,
        result,
        save_path
    ):

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            save_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)
            writer.writerow(
                [
                    "Metric",
                    "Value"
                ]
            )

            for k, v in result.items():
                writer.writerow(
                    [
                        k,
                        v
                    ]
                )

    # ======================================================
    # PRINT
    # ======================================================

    def print_summary(
        self,
        result
    ):

        print("=" * 50)
        print("Evaluation Summary")
        print("=" * 50)

        for k, v in result.items():
            print(f"{k:30}: {v}")

        print("=" * 50)