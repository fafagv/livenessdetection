"""
finger_counter.py — Open finger state & counting analysis.

Refactored from FingerCountingProject.py. Now delegates the actual
finger-up detection to GestureDetector/HandDetector instead of
re-implementing the tip-comparison logic locally (it was duplicated
verbatim in the original file).
"""

import logging
import os

import cv2

from .gesture_detector import GestureDetector

logger = logging.getLogger(__name__)


def load_overlay_images(folder_path: str):
    """Load numbered overlay images (1.png..5.png) used to display the finger count."""
    if not os.path.isdir(folder_path):
        logger.warning("Overlay folder '%s' not found — running without overlays.", folder_path)
        return []
    overlay_list = []
    for img_path in sorted(os.listdir(folder_path)):
        image = cv2.imread(os.path.join(folder_path, img_path))
        if image is not None:
            overlay_list.append(image)
    return overlay_list


def run(camera_index: int = 0, overlay_folder: str = "data/resources/finger_images", cam_size=(640, 480)):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return

    cap.set(3, cam_size[0])
    cap.set(4, cam_size[1])

    overlay_list = load_overlay_images(overlay_folder)
    detector = GestureDetector(max_hands=1, detection_con=0.75)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, state = detector.analyze(img, draw=False)

            if state:
                total_fingers = state.fingers_count
                logger.info("Fingers up: %d", total_fingers)

                if overlay_list and 0 <= total_fingers - 1 < len(overlay_list):
                    h, w, _ = overlay_list[total_fingers - 1].shape
                    img[0:h, 0:w] = overlay_list[total_fingers - 1]

                cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    img, str(total_fingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25,
                )

            cv2.imshow("Finger Counter", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
