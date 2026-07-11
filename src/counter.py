from __future__ import annotations


class VehicleCounter:
    """
    Count detected vehicles and wrong-way vehicles.

    Each vehicle is counted only once using its track ID.
    """

    def __init__(self):

        # All counted vehicle IDs
        self.vehicle_ids = set()

        # Wrong-way vehicle IDs
        self.wrong_way_ids = set()

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        track_id: int,
        wrong_way: bool = False
    ):
        """
        Update counters.

        Parameters
        ----------
        track_id : int
            Vehicle ID.

        wrong_way : bool
            Whether the vehicle is wrong-way.
        """

        self.vehicle_ids.add(track_id)

        if wrong_way:
            self.wrong_way_ids.add(track_id)

    # =====================================================
    # Total Vehicle
    # =====================================================

    @property
    def total_vehicle(self) -> int:

        return len(self.vehicle_ids)

    # =====================================================
    # Wrong-way Vehicle
    # =====================================================

    @property
    def total_wrong_way(self) -> int:

        return len(self.wrong_way_ids)

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.vehicle_ids.clear()
        self.wrong_way_ids.clear()

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        return {
            "total_vehicle": self.total_vehicle,
            "total_wrong_way": self.total_wrong_way
        }