from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Tuple, List

from src.utils import bottom_center


class DirectionDetector:
    """
    Estimate vehicle direction using bottom-center trajectory.
    """

    def __init__(

        self,

        history_size: int = 20,

        movement_threshold: int = 10,

        allowed_direction: str = "DOWN"

    ):
        """
        Parameters
        ----------
        history_size :
            Maximum stored trajectory points.

        movement_threshold :
            Ignore very small movements.

        allowed_direction :
            Vehicle legal direction.

            UP
            DOWN
            LEFT
            RIGHT
        """

        self.history_size = history_size

        self.threshold = movement_threshold

        self.allowed_direction = allowed_direction.upper()

        self.history = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    # =====================================================
    # Update
    # =====================================================

    def update(

        self,

        track_id: int,

        bbox

    ):

        """
        Update trajectory of one vehicle.
        """

        point = bottom_center(bbox)

        self.history[track_id].append(point)

    # =====================================================
    # Get trajectory
    # =====================================================

    def get_trajectory(

        self,

        track_id

    ) -> List[Tuple[int, int]]:

        return list(

            self.history.get(track_id, [])

        )

    # =====================================================
    # Direction
    # =====================================================

    def get_direction(

        self,

        track_id

    ) -> str:

        points = self.history.get(track_id)

        if points is None:

            return "UNKNOWN"

        if len(points) < 2:

            return "UNKNOWN"

        x0, y0 = points[0]

        x1, y1 = points[-1]

        dx = x1 - x0

        dy = y1 - y0

        if abs(dx) < self.threshold and abs(dy) < self.threshold:

            return "STATIONARY"

        if abs(dx) > abs(dy):

            if dx > 0:

                return "RIGHT"

            else:

                return "LEFT"

        else:

            if dy > 0:

                return "DOWN"

            else:

                return "UP"

    # =====================================================
    # Wrong-way
    # =====================================================

    def is_wrong_way(

        self,

        track_id

    ) -> bool:

        direction = self.get_direction(track_id)

        if direction in [

            "UNKNOWN",

            "STATIONARY"

        ]:

            return False

        return direction != self.allowed_direction

    # =====================================================
    # Remove lost ID
    # =====================================================

    def remove(

        self,

        track_id

    ):

        if track_id in self.history:

            del self.history[track_id]

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.history.clear()