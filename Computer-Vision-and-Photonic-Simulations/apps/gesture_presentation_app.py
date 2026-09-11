"""
gesture_presentation_app.py — Main application entry point for gesture-controlled
slide presentation.

Refactored from main3 (2).py (the more complete of two near-duplicate
presentation scripts in the original project — main3 (1).py was an earlier,
much simpler draft that only listed image paths and was dropped).

Fixes from the original:
    - Guards against an empty/missing presentation_slides folder instead of
      raising an IndexError on pathImages[0].
    - Proper camera release / window cleanup via try/finally.
    - 'q' *or* ESC both exit, matching the rest of this repo's apps.

Gestures (hand raised above the green threshold line):
    Thumb only up   -> previous slide
    Pinky only up   -> next slide
Gestures (hand below the threshold line):
    Index+middle up -> pointer (draws a dot, doesn't persist)
    Index only up   -> draw/annotate
    Index+middle+ring up -> undo last annotation

Run:
    python -m apps.gesture_presentation_app
"""

import logging
import os

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 1280, 720
GESTURE_THRESHOLD = 300


def run(slides_folder: str = "data/presentation_slides", camera_index: int = 0):
    if not os.path.isdir(slides_folder) or not os.listdir(slides_folder):
        logger.error("No presentation slides found in '%s'.", slides_folder)
        return

    path_images = sorted(os.listdir(slides_folder), key=len)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    detector_hand = HandDetector(detectionCon=0.8, maxHands=1)

    delay = 30
    button_pressed = False
    counter = 0
    img_number = 0
    annotations = [[]]
    annotation_number = -1
    annotation_start = False
    hs, ws = 120, 213

    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)

            path_full_image = os.path.join(slides_folder, path_images[img_number])
            img_current = cv2.imread(path_full_image)
            if img_current is None:
                logger.warning("Could not read slide image '%s' — skipping.", path_full_image)
                img_number = min(img_number + 1, len(path_images) - 1)
                continue

            hands, img = detector_hand.findHands(img)
            cv2.line(img, (0, GESTURE_THRESHOLD), (WIDTH, GESTURE_THRESHOLD), (0, 255, 0), 10)

            if hands and not button_pressed:
                hand = hands[0]
                cx, cy = hand["center"]
                lm_list = hand["lmList"]
                fingers = detector_hand.fingersUp(hand)

                x_val = int(np.interp(lm_list[8][0], [WIDTH // 2, WIDTH], [0, WIDTH]))
                y_val = int(np.interp(lm_list[8][1], [150, HEIGHT - 150], [0, HEIGHT]))
                index_finger = x_val, y_val

                if cy <= GESTURE_THRESHOLD:
                    if fingers == [1, 0, 0, 0, 0]:
                        button_pressed = True
                        if img_number > 0:
                            img_number -= 1
                            annotations = [[]]
                            annotation_number = -1
                            annotation_start = False
                    if fingers == [0, 0, 0, 0, 1]:
                        button_pressed = True
                        if img_number < len(path_images) - 1:
                            img_number += 1
                            annotations = [[]]
                            annotation_number = -1
                            annotation_start = False

                if fingers == [0, 1, 1, 0, 0]:
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)

                if fingers == [0, 1, 0, 0, 0]:
                    if not annotation_start:
                        annotation_start = True
                        annotation_number += 1
                        annotations.append([])
                    annotations[annotation_number].append(index_finger)
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)
                else:
                    annotation_start = False

                if fingers == [0, 1, 1, 1, 0]:
                    if annotations:
                        annotations.pop(-1)
                        annotation_number -= 1
                        button_pressed = True
            else:
                annotation_start = False

            if button_pressed:
                counter += 1
                if counter > delay:
                    counter = 0
                    button_pressed = False

            for annotation in annotations:
                for j in range(1, len(annotation)):
                    cv2.line(img_current, annotation[j - 1], annotation[j], (0, 0, 200), 12)

            img_small = cv2.resize(img, (ws, hs))
            h, w, _ = img_current.shape
            img_current[0:hs, w - ws:w] = img_small

            cv2.imshow("Slides", img_current)
            cv2.imshow("Camera", img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
