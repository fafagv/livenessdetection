"""
virtual_zoom.py — Multi-touch / dual-hand image scaling.

Refactored from VirtualZoom.py. This module relies on cvzone's
HandTrackingModule (an external package, see requirements.txt) rather than
this repo's own hand_tracker.HandDetector, because the original two-hand
"pinch to zoom" gesture depends on cvzone's per-hand dict API
(fingersUp(hand), hand["center"], hand["lmList"]) which our single-hand
HandDetector does not replicate. dataCollection.py, which was an earlier,
less complete duplicate of the same capture pattern, was dropped in favor
of this version.

Fixes from the original:
    - Replaced the bare `except: pass` around the resize/overlay block with
      a narrow try/except that logs unexpected failures.
    - Guards for a missing overlay image instead of silently doing nothing
      forever.
    - Proper camera release / window cleanup on exit.
"""

import logging

import cv2
from cvzone.HandTrackingModule import HandDetector

logger = logging.getLogger(__name__)


def run(overlay_image_path: str = "data/resources/zoom_overlay.jpg", camera_index: int = 0, cam_size=(1280, 720)):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, cam_size[0])
    cap.set(4, cam_size[1])

    overlay_img = cv2.imread(overlay_image_path)
    if overlay_img is None:
        logger.warning("Overlay image '%s' not found — zoom will have nothing to render.", overlay_image_path)

    detector = HandDetector(detectionCon=0.7)
    start_dist = None
    scale = 0
    cx, cy = cam_size[0] // 2, cam_size[1] // 2

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            hands, img = detector.findHands(img)

            if len(hands) == 2:
                if (
                    detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0]
                    and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]
                ):
                    length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                    if start_dist is None:
                        start_dist = length

                    scale = int((length - start_dist) // 2)
                    cx, cy = info[4:]
            else:
                start_dist = None

            if overlay_img is not None:
                try:
                    h1, w1, _ = overlay_img.shape
                    new_h, new_w = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
                    if new_h > 0 and new_w > 0:
                        resized = cv2.resize(overlay_img, (new_w, new_h))
                        y1, y2 = cy - new_h // 2, cy + new_h // 2
                        x1, x2 = cx - new_w // 2, cx + new_w // 2
                        if 0 <= y1 and y2 <= img.shape[0] and 0 <= x1 and x2 <= img.shape[1]:
                            img[y1:y2, x1:x2] = resized
                except Exception:
                    logger.exception("Failed to composite zoomed overlay onto frame.")

            cv2.imshow("Virtual Zoom", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
