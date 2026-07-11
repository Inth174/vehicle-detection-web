from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


# ==========================================================
# Timer
# ==========================================================

class Timer:
    """
    Measure processing time.
    """

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self) -> float:

        if self.start_time is None:
            return 0.0

        return time.perf_counter() - self.start_time


# ==========================================================
# Folder
# ==========================================================

def ensure_dir(path: str | Path):

    """
    Create folder if it does not exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# Image
# ==========================================================

def read_image(image_path: str):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

    return img


def save_image(save_path: str, image):

    ensure_dir(Path(save_path).parent)

    cv2.imwrite(save_path, image)


# ==========================================================
# Video
# ==========================================================

def open_video(video_path: str):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    return cap


def create_video_writer(
    save_path: str,
    fps: float,
    width: int,
    height: int
):

    ensure_dir(Path(save_path).parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        fps,
        (width, height)
    )

    return writer


# ==========================================================
# Geometry
# ==========================================================

def bbox_center(box):

    """
    box = (x1,y1,x2,y2)
    """

    x1, y1, x2, y2 = box

    return (

        int((x1 + x2) / 2),

        int((y1 + y2) / 2)

    )


def bottom_center(box):

    """
    Bottom-center point.

    Used for direction estimation.
    """

    x1, y1, x2, y2 = box

    return (

        int((x1 + x2) / 2),

        int(y2)

    )


def bbox_area(box):

    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


# ==========================================================
# Color
# ==========================================================

GREEN = (0,255,0)

RED = (0,0,255)

BLUE = (255,0,0)

YELLOW = (0,255,255)

WHITE = (255,255,255)

BLACK = (0,0,0)


def random_color(track_id: int):

    np.random.seed(track_id)

    return tuple(
        int(x)
        for x in np.random.randint(0,255,3)
    )


# ==========================================================
# Text
# ==========================================================

def put_text(

    image,

    text,

    position,

    color=GREEN,

    scale=0.6,

    thickness=2

):

    cv2.putText(

        image,

        text,

        position,

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        color,

        thickness,

        cv2.LINE_AA

    )


# ==========================================================
# FPS
# ==========================================================

class FPSCounter:

    """
    Real-time FPS counter.
    """

    def __init__(self):

        self.prev_time = time.time()

        self.fps = 0

    def update(self):

        now = time.time()

        dt = now - self.prev_time

        self.prev_time = now

        if dt > 0:

            self.fps = 1 / dt

        return self.fps


# ==========================================================
# Misc
# ==========================================================

def format_time(seconds: float):

    return f"{seconds:.3f} s"


def file_exists(path: str):

    return os.path.exists(path)