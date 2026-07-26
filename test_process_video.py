#!/usr/bin/env python3
"""Checks that process_video stops decoding once a label is confirmed.

Uses a stub model so it needs no weights and no GPU. Run: .venv/bin/python test_process_video.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

import filter_camera as fc

TOTAL_FRAMES = 300  # 10s @ 30fps -> 10 sampled frames at VIDEO_SAMPLE_INTERVAL=30


class Box:
    def __init__(self, cls_id, conf):
        self.cls, self.conf = [cls_id], [conf]
        self.xyxy = [(0, 0, 960, 540)]  # half the frame, well above MIN_BOX_AREA_RATIO


class Results:
    def __init__(self, boxes):
        self.orig_shape, self.boxes = (1080, 1920), boxes


class StubModel:
    """Returns `boxes_per_frame` for every frame and counts how many it saw."""

    def __init__(self, boxes_per_frame):
        self.boxes_per_frame, self.frames_seen = boxes_per_frame, 0

    def __call__(self, batch, **kwargs):
        self.frames_seen += len(batch)
        return [Results(self.boxes_per_frame) for _ in batch]


def make_video(path):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype="uint8")
    for _ in range(TOTAL_FRAMES):
        w.write(frame)
    w.release()


def main():
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "IMAG0001.AVI"
    make_video(video)

    sampled = sum(1 for _ in fc.iter_sampled_frames(video))
    assert sampled == TOTAL_FRAMES / fc.VIDEO_SAMPLE_INTERVAL, f"sampled {sampled} frames"

    # A person in every frame: confirmed within the first batch, rest never decoded.
    model = StubModel([Box(1, 0.9)])
    detected = fc.process_video(model, video, fc.WANTED_CLASSES, fc.CONFIDENCE_THRESHOLD)
    assert detected == ["person(90%)"], detected
    assert model.frames_seen == fc.BATCH_SIZE, f"no early exit: saw {model.frames_seen} frames"
    assert model.frames_seen < sampled, "early exit must skip frames"

    # Nothing detectable: full scan, no false positive.
    model = StubModel([])
    assert fc.process_video(model, video, fc.WANTED_CLASSES, fc.CONFIDENCE_THRESHOLD) == []
    assert model.frames_seen == sampled, f"full scan expected, saw {model.frames_seen}"

    # Animal only (class 0) is not a wanted class.
    model = StubModel([Box(0, 0.9)])
    assert fc.process_video(model, video, fc.WANTED_CLASSES, fc.CONFIDENCE_THRESHOLD) == []

    # A box too small to matter is ignored even though the class is wanted.
    tiny = Box(1, 0.9)
    tiny.xyxy = [(0, 0, 40, 40)]
    model = StubModel([tiny])
    assert fc.process_video(model, video, fc.WANTED_CLASSES, fc.CONFIDENCE_THRESHOLD) == []

    print("ok")


if __name__ == "__main__":
    main()
