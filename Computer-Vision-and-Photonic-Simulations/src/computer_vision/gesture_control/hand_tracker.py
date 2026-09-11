"""
hand_tracker.py — Core MediaPipe hand tracking class.

Refactored from HandTrackingModule.py (the most complete of four near-duplicate
copies found in the original project: HandTrackingModule.py, hand_tracking_module.py,
"HandTrackingModule (2).py", and Basic.py — all consolidated into this single module).

Fixes from the original:
    - Updated to the current mediapipe.solutions.hands API (unchanged signature,
      but pinned in requirements.txt to a version where it's still supported —
      MediaPipe deprecated the legacy `solutions` API in favor of the Tasks API
      in newer releases; pin mediapipe<0.10.14 or migrate to
      mediapipe.tasks.python.vision.HandLandmarker for newest versions).
    - Added a camera-release/cleanup path in the __main__ demo.
    - Guarded findDistance()/fingersUp() against being called before any hand
      has been detected (original raised IndexError on empty self.lmList).
"""

import math
import time
import logging

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class HandDetector:
    """Wraps MediaPipe Hands to detect landmarks, positions, fingers-up state, and distances."""

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.lm_list = []

    def find_hands(self, img, draw: bool = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no: int = 0, draw: bool = True):
        x_list, y_list, bbox = [], [], []
        self.lm_list = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_no >= len(self.results.multi_hand_landmarks):
                return self.lm_list, bbox

            my_hand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(my_hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        """Return a list of 5 ints (1=up, 0=down) for [thumb, index, middle, ring, pinky]."""
        if not self.lm_list:
            logger.warning("fingers_up() called with no hand detected yet.")
            return [0, 0, 0, 0, 0]

        fingers = []
        # Thumb (compares x, since thumb moves sideways)
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers (compare y)
        for lm_id in range(1, 5):
            if self.lm_list[self.tip_ids[lm_id]][2] < self.lm_list[self.tip_ids[lm_id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True, r=15, t=3):
        if not self.lm_list or p1 >= len(self.lm_list) or p2 >= len(self.lm_list):
            raise ValueError("find_distance() called with invalid landmark indices / no hand detected.")

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def _demo():
    """Standalone webcam demo — mirrors the original module's __main__ block."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    detector = HandDetector()
    p_time = 0
    try:
        while True:
            success, img = cap.read()
            if not success:
                logger.warning("Failed to read frame from webcam.")
                break

            img = detector.find_hands(img)
            lm_list, _ = detector.find_position(img)
            if lm_list:
                print(lm_list[4])

            c_time = time.time()
            fps = 1 / (c_time - p_time) if c_time != p_time else 0
            p_time = c_time

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
