"""
gesture_games_app.py — Interactive gesture game & hand-distance estimation.

Refactored from Game.py. A "balloon pop" style game: bring your hand within
~40cm of the camera (estimated from the pixel span between two knuckle
landmarks, calibrated with a quadratic fit) while your palm overlaps a
randomly-placed target circle to score a point.

Bug fixed: the original had
    `hands = detector.findHands(img, draw=False)`
instead of
    `hands, img = detector.findHands(img, draw=False)`.
cvzone's findHands() returns a (hands, img) tuple; assigning it to a single
variable meant `hands` was actually that whole tuple, `if hands:` was always
truthy (a 2-tuple is never falsy), and `hands[0]['lmList']` would have
raised a TypeError the first time a hand appeared, since hands[0] was really
the hands-list, not a per-hand dict. This crashed the original script as
soon as a hand was shown to the camera.

Run:
    python -m apps.gesture_games_app
"""

import logging
import math
import random
import time

import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

logger = logging.getLogger(__name__)

# Calibration table: x = raw pixel distance between two knuckle landmarks,
# y = corresponding real-world distance in cm.
_CALIBRATION_X = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
_CALIBRATION_Y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def run(camera_index: int = 0, total_time: int = 20):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    coeffs = np.polyfit(_CALIBRATION_X, _CALIBRATION_Y, 2)  # y = A x^2 + B x + C

    cx, cy = 250, 250
    color = (255, 0, 255)
    counter = 0
    score = 0
    time_start = time.time()

    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)

            if time.time() - time_start < total_time:
                hands, img = detector.findHands(img, draw=False)

                if hands:
                    lm_list = hands[0]["lmList"]
                    x, y, w, h = hands[0]["bbox"]
                    x1, y1 = lm_list[5]
                    x2, y2 = lm_list[17]

                    distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                    a, b, c = coeffs
                    distance_cm = a * distance ** 2 + b * distance + c

                    if distance_cm < 40 and x < cx < x + w and y < cy < y + h:
                        counter = 1

                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                    cvzone.putTextRect(img, f"{int(distance_cm)} cm", (x + 5, y - 10))

                if counter:
                    counter += 1
                    color = (0, 255, 0)
                    if counter == 3:
                        cx = random.randint(100, 1100)
                        cy = random.randint(100, 600)
                        color = (255, 0, 255)
                        score += 1
                        counter = 0

                cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
                cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)
                cv2.circle(img, (cx, cy), 30, (50, 50, 50), 2)

                cvzone.putTextRect(img, f"Time: {int(total_time - (time.time() - time_start))}",
                                    (1000, 75), scale=3, offset=20)
                cvzone.putTextRect(img, f"Score: {str(score).zfill(2)}", (60, 75), scale=3, offset=20)
            else:
                cvzone.putTextRect(img, "Game Over", (400, 400), scale=5, offset=30, thickness=7)
                cvzone.putTextRect(img, f"Your Score: {score}", (450, 500), scale=3, offset=20)
                cvzone.putTextRect(img, "Press R to restart, ESC to quit", (400, 575), scale=2, offset=10)

            cv2.imshow("Gesture Game", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                time_start = time.time()
                score = 0
            elif key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
