"""
==============================================================
tracker.py
DeepSORT / ByteTrack
==============================================================
"""

from deep_sort_realtime.deepsort_tracker import DeepSort

# supervision 0.26.x
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision import Detections


class MultiObjectTracker:

    def __init__(
        self,
        tracker_type="deepsort"
    ):

        self.tracker_type = tracker_type.lower()

        if self.tracker_type == "deepsort":
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7
            )

        elif self.tracker_type == "bytetrack":
            self.tracker = ByteTrack()

        else:
            raise ValueError(
                "tracker_type must be 'deepsort' or 'bytetrack'"
            )

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        detections,
        frame=None
    ):

        outputs = []

        # -------------------------------------------------
        # DeepSORT
        # -------------------------------------------------

        if self.tracker_type == "deepsort":

            ds_boxes = []

            for det in detections:
                xmin, ymin, xmax, ymax = det["bbox"]

                ds_boxes.append(
                    (
                        [xmin, ymin, xmax - xmin, ymax - ymin],
                        det["confidence"],
                        det["class_id"]
                    )
                )

            tracks = self.tracker.update_tracks(
                ds_boxes,
                frame=frame
            )

            for track in tracks:
                if not track.is_confirmed():
                    continue

                l, t, r, b = track.to_ltrb()
                best_det = None
                best_iou = -1

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]

                    xx1 = max(l, x1)
                    yy1 = max(t, y1)
                    xx2 = min(r, x2)
                    yy2 = min(b, y2)

                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

                    area_track = max(0, r - l) * max(0, b - t)
                    area_det = max(0, x2 - x1) * max(0, y2 - y1)

                    union = area_track + area_det - inter + 1e-6
                    iou = inter / union

                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                if best_det is None:
                    continue

                outputs.append({
                    "track_id": int(track.track_id),
                    "bbox": [
                        int(l),
                        int(t),
                        int(r),
                        int(b)
                    ],
                    "class_id": best_det["class_id"],
                    "class_name": best_det["class_name"],
                    "confidence": best_det["confidence"]
                })

        # -------------------------------------------------
        # ByteTrack
        # -------------------------------------------------

        else:
            import numpy as np

            xyxy = []
            conf = []
            cls = []

            for det in detections:
                xyxy.append(det["bbox"])
                conf.append(det["confidence"])
                cls.append(det["class_id"])

            # Không có detection
            if len(xyxy) == 0:
                return []

            detections_sv = Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(conf, dtype=np.float32),
                class_id=np.array(cls, dtype=np.int32),
            )

            tracks = self.tracker.update_with_detections(
                detections_sv
            )

            for i in range(len(tracks)):
                outputs.append({
                    "track_id": int(
                        tracks.tracker_id[i]
                    ),
                    "bbox": tracks.xyxy[i].astype(int).tolist(),
                    
                    "class_id": detections[i]["class_id"],
                    "class_name": detections[i]["class_name"],
                    "confidence": detections[i]["confidence"]
                })
        return outputs
