"""
model_tester.py — Keras/MediaPipe sign-recognition model evaluation.

Refactored from Testing.py. Reuses crop_hand_to_square() from
data_collection.py instead of duplicating the crop/resize logic.

Fixes from the original:
    - Guards model/labels file paths and gives a clear error if missing,
      instead of an opaque exception from cvzone.ClassificationModule.
    - `labels` list length is validated against the classifier's predicted
      index range to avoid an IndexError on a mismatched label file.
    - Camera is released on exit.
"""

import logging
import os

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

from .data_collection import crop_hand_to_square

logger = logging.getLogger(__name__)


def run(
    model_path: str = "data/resources/sign_language_model/keras_model.h5",
    labels_path: str = "data/resources/sign_language_model/labels.txt",
    labels=("A", "B", "C"),
    camera_index: int = 0,
    img_size: int = 300,
    offset: int = 20,
):
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        logger.error("Model/labels not found at '%s' / '%s'. Train or export them first.", model_path, labels_path)
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return

    detector = HandDetector(maxHands=1)
    classifier = Classifier(model_path, labels_path)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img_output = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                bbox = hands[0]["bbox"]
                x, y, w, h = bbox
                img_white, img_crop = crop_hand_to_square(img, bbox, offset, img_size)

                if img_crop.size == 0:
                    continue

                prediction, index = classifier.getPrediction(img_white, draw=False)
                label = labels[index] if index < len(labels) else "?"

                cv2.rectangle(
                    img_output, (x - offset, y - offset - 50),
                    (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED,
                )
                cv2.putText(img_output, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", img_crop)
                cv2.imshow("ImageWhite", img_white)

            cv2.imshow("Image", img_output)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
