from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    Standard detection object shared across the project.

    Attributes
    ----------
    bbox : tuple
        Bounding box (x1, y1, x2, y2)

    confidence : float
        Detection confidence.

    class_id : int
        YOLO class id.

    class_name : str
        Vehicle class name.

    track_id : Optional[int]
        Tracker ID.
        None when tracking is not enabled.

    direction : Optional[str]
        Vehicle movement direction.

    wrong_way : bool
        Whether the vehicle is moving in the wrong direction.
    """

    bbox: Tuple[int, int, int, int]

    confidence: float

    class_id: int

    class_name: str

    track_id: Optional[int] = None

    direction: Optional[str] = None

    wrong_way: bool = False