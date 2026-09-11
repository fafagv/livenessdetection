"""
gesture_detector.py — Spatial hand gesture recognition engine.

The original project re-implemented the same three primitives — "how many
fingers are up", "what's the pinch distance between two landmarks", and
"is the hand within a usable bounding-box area" — separately inside
Gesture.py, FingerCountingProject.py, and VirtualZoom.py. This module
consolidates that duplicated logic into one reusable engine built on top
of hand_tracker.HandDetector, and Project.py's minimal landmark-printing
loop is folded into the __main__ demo below.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .hand_tracker import HandDetector

logger = logging.getLogger(__name__)


@dataclass
class GestureState:
    lm_list: list
    bbox: tuple
    fingers: List[int]
    fingers_count: int
    pinch_distance: Optional[float]


class GestureDetector:
    """High-level gesture engine: wraps HandDetector with common gesture primitives."""

    def __init__(self, max_hands: int = 1, detection_con: float = 0.7):
        self.hand_detector = HandDetector(max_hands=max_hands, detection_con=detection_con)

    def analyze(self, img, draw: bool = True) -> Tuple[object, Optional[GestureState]]:
        """Run hand detection on a frame and return (annotated_img, GestureState|None)."""
        img = self.hand_detector.find_hands(img, draw=draw)
        lm_list, bbox = self.hand_detector.find_position(img, draw=draw)

        if not lm_list:
            return img, None

        fingers = self.hand_detector.fingers_up()
        pinch_distance = None
        if len(lm_list) > 8:
            pinch_distance, img, _ = self.hand_detector.find_distance(4, 8, img, draw=draw)

        state = GestureState(
            lm_list=lm_list,
            bbox=bbox,
            fingers=fingers,
            fingers_count=fingers.count(1),
            pinch_distance=pinch_distance,
        )
        return img, state

    @staticmethod
    def bbox_area(bbox: tuple) -> float:
        """Area of the hand's bounding box, scaled down like the original scripts did (//100)."""
        if not bbox:
            return 0.0
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 100

    @staticmethod
    def map_range(value: float, in_range: Tuple[float, float], out_range: Tuple[float, float]) -> float:
        """np.interp wrapper, matching the [50, 300] -> [0, 100]-style mappings used across the app."""
        return float(np.interp(value, in_range, out_range))


def _demo():
    import cv2
    import time

    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    p_time = 0
    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, state = detector.analyze(img)
            if state:
                logger.info("Fingers up: %d, pinch distance: %s", state.fingers_count, state.pinch_distance)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if c_time != p_time else 0
            p_time = c_time
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv2.imshow("Gesture Detector", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
