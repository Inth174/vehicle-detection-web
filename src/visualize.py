from __future__ import annotations

import cv2
import numpy as np

from src.utils import (
    GREEN,
    RED,
    BLUE,
    WHITE,
    BLACK,
    YELLOW,
    random_color,
    put_text,
)


# ==========================================================
# Bounding Box
# ==========================================================

def draw_bbox(
    image,
    box,
    color=GREEN,
    thickness=2
):
    """
    Draw bounding box.

    Parameters
    ----------
    image : ndarray
    box : (x1,y1,x2,y2)
    """

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


# ==========================================================
# Track ID
# ==========================================================

def draw_track_id(
    image,
    box,
    track_id,
    wrong_way=False
):

    x1, y1, _, _ = map(int, box)

    if wrong_way:
        text = f"ID {track_id} | WRONG WAY"
        color = RED
    else:
        text = f"ID {track_id}"
        color = WHITE

    put_text(
        image,
        text,
        (x1, max(20, y1 - 10)),
        color
    )


# ==========================================================
# Vehicle Class
# ==========================================================

def draw_class(
    image,
    box,
    class_name,
):

    x1, y1, _, _ = map(int, box)

    put_text(
        image,
        class_name,
        (x1, max(20, y1 - 30)),
        YELLOW
    )

# ==========================================================
# Confidence
# ==========================================================

def draw_confidence(
    image,
    box,
    confidence
):

    x1, y1, _, _ = map(int, box)

    put_text(
        image,
        f"{confidence*100:.1f}%",
        (x1, max(40, y1 - 50)),
        GREEN
    )

# ==========================================================
# Direction
# ==========================================================

def draw_direction(
    image,
    box,
    direction,
    wrong_way=False
):

    _, _, _, y2 = map(int, box)

    if wrong_way:
        text = f"{direction} (WRONG)"
        color = RED
    else:
        text = direction
        color = BLUE

    put_text(
        image,
        text,
        (int(box[0]), y2 + 20),
        color
    )

# ==========================================================
# Trajectory
# ==========================================================

def draw_trajectory(
    image,
    points,
    color=BLUE
):

    if len(points) < 2:
        return

    for i in range(1, len(points)):

        cv2.line(
            image,
            points[i-1],
            points[i],
            color,
            2
        )


# ==========================================================
# FPS
# ==========================================================

def draw_fps(
    image,
    fps
):

    put_text(
        image,
        f"FPS : {fps:.2f}",
        (20, 30),
        GREEN
    )


# ==========================================================
# Processing Time
# ==========================================================

def draw_processing_time(
    image,
    processing_time
):

    put_text(
        image,
        f"Time : {processing_time:.3f}s",
        (20, 60),
        GREEN
    )


# ==========================================================
# Vehicle Counter
# ==========================================================

def draw_vehicle_count(
    image,
    count
):

    put_text(
        image,
        f"Vehicles : {count}",
        (20, 90),
        YELLOW
    )


# ==========================================================
# Wrong-way Counter
# ==========================================================

def draw_wrong_way_count(
    image,
    count
):

    put_text(
        image,
        f"Wrong-way : {count}",
        (20, 120),
        RED
    )


# ==========================================================
# Detection Line
# ==========================================================

def draw_line(
    image,
    pt1,
    pt2,
    color=BLUE,
    thickness=2
):

    cv2.line(
        image,
        pt1,
        pt2,
        color,
        thickness
    )


# ==========================================================
# Detection Region
# ==========================================================

def draw_polygon(
    image,
    polygon,
    color=BLUE
):

    polygon = np.array(
        polygon,
        np.int32
    )

    cv2.polylines(
        image,
        [polygon],
        True,
        color,
        2
    )


# ==========================================================
# Complete Vehicle Annotation
# ==========================================================

def draw_vehicle(
    image,
    box,
    track_id=None,
    class_name=None,
    confidence=None,
    direction=None,
    wrong_way=False,
):

    color = RED if wrong_way else random_color(
        track_id if track_id is not None else 0
    )

    draw_bbox(
        image,
        box,
        color
    )

    if track_id is not None:

        draw_track_id(
            image,
            box,
            track_id,
            wrong_way
        )

    if class_name is not None:

        draw_class(
            image,
            box,
            class_name
        )

    if confidence is not None:

        draw_confidence(
            image,
            box,
            confidence
        )

    if direction is not None:

        draw_direction(
            image,
            box,
            direction,
            wrong_way
        )


# ==========================================================
# Final Overlay
# ==========================================================

def draw_dashboard(
    image,
    fps,
    processing_time,
    vehicle_count,
    wrong_way_count,
):

    cv2.rectangle(
        image,
        (10, 10),
        (290, 140),
        (35, 35, 35),
        -1
    )

    draw_fps(
        image,
        fps
    )

    draw_processing_time(
        image,
        processing_time
    )

    draw_vehicle_count(
        image,
        vehicle_count
    )

    draw_wrong_way_count(
        image,
        wrong_way_count
    )
