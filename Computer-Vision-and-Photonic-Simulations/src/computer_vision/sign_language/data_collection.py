"""
data_collection.py — Landmark recording pipeline for sign-language training data.

Refactored from DataCollection1.py (dataCollection.py, an earlier and less
complete duplicate that saved un-padded crops with no aspect-ratio handling,
was dropped).

Fixes from the original:
    - Guards the crop region against going out of frame bounds (the original
      would raise on hands near the frame edge, since `y - offset` etc. can
      go negative or exceed the frame).
    - Ensures the output folder exists instead of crashing on cv2.imwrite.
    - Wrapped in a function with camera cleanup instead of an unguarded
      infinite loop.
"""

import logging
import math
import os
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

logger = logging.getLogger(__name__)


def crop_hand_to_square(img, bbox, offset: int = 20, img_size: int = 300):
    """
    Crop the hand region (with padding) out of `img` and paste it, centered
    and aspect-ratio-preserved, onto a fixed img_size x img_size white canvas.
    Shared by data_collection.py and model_tester.py to avoid duplicating
    this logic (it was copy-pasted between DataCollection1.py and Testing.py
    in the original project).
    """
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]

    y1, y2 = max(0, y - offset), min(img_h, y + h + offset)
    x1, x2 = max(0, x - offset), min(img_w, x + w + offset)
    img_crop = img[y1:y2, x1:x2]

    img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
    if img_crop.size == 0:
        return img_white, img_crop

    aspect_ratio = h / w if w else 1
    try:
        if aspect_ratio > 1:
            k = img_size / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (max(1, w_cal), img_size))
            w_gap = math.ceil((img_size - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_resize
        else:
            k = img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (img_size, max(1, h_cal)))
            h_gap = math.ceil((img_size - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize
    except Exception:
        logger.exception("Failed to resize/paste hand crop — returning blank canvas.")

    return img_white, img_crop


def run(output_folder: str = "data/images/sign_language/C", camera_index: int = 0, img_size: int = 300, offset: int = 20):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return

    detector = HandDetector(maxHands=1)
    counter = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            hands, img = detector.findHands(img)
            img_white = None
            if hands:
                img_white, img_crop = crop_hand_to_square(img, hands[0]["bbox"], offset, img_size)
                cv2.imshow("ImageCrop", img_crop)
                cv2.imshow("ImageWhite", img_white)

            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and img_white is not None:
                counter += 1
                cv2.imwrite(f"{output_folder}/Image_{time.time()}.jpg", img_white)
                logger.info("Saved sample #%d", counter)
            elif key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
