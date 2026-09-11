"""
encoder.py — Facial feature extraction & embedding generation.

Refactored from EncodeGenerator.py. Scans a folder of student photos,
computes a face_recognition embedding for each, optionally uploads the
source photos to Firebase Storage, and pickles the (encodings, ids) pair
for later use by tracker.py / apps/face_attendance_app.py.

Fixes from the original:
    - Firebase init moved to db_handler.init_firebase() (no hardcoded creds).
    - Guards against images with zero detected faces (original crashed with
      an IndexError on face_recognition.face_encodings(img)[0]).
    - Wrapped in functions with proper exception handling instead of running
      top-level script logic with no error recovery.
"""

import os
import pickle
import logging
from typing import List, Tuple

import cv2
import face_recognition

from .db_handler import init_firebase, get_storage_bucket

logger = logging.getLogger(__name__)


def load_student_images(folder_path: str) -> Tuple[List, List[str]]:
    """Load every image in folder_path, returning (images, student_ids)."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    img_list, student_ids = [], []
    for path in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, path)
        img = cv2.imread(full_path)
        if img is None:
            logger.warning("Skipping unreadable image: %s", full_path)
            continue
        img_list.append(img)
        student_ids.append(os.path.splitext(path)[0])

    return img_list, student_ids


def upload_images_to_storage(folder_path: str, filenames: List[str]) -> None:
    """Upload each image to the Firebase Storage bucket (best-effort)."""
    bucket = get_storage_bucket()
    for path in filenames:
        file_path = os.path.join(folder_path, path)
        try:
            blob = bucket.blob(file_path)
            blob.upload_from_filename(file_path)
        except Exception:
            logger.exception("Failed to upload %s to storage", file_path)


def find_encodings(images_list: List) -> List:
    """Compute one face-recognition embedding per image, skipping faceless images."""
    encode_list = []
    for img in images_list:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if not encodings:
                logger.warning("No face detected in one of the images — skipping.")
                continue
            encode_list.append(encodings[0])
        except Exception:
            logger.exception("Failed to encode an image — skipping.")
    return encode_list


def generate_encode_file(
    folder_path: str = "data/images",
    output_path: str = "EncodeFile.p",
    upload_to_storage: bool = False,
) -> None:
    """Full pipeline: load images -> (optional upload) -> encode -> pickle to disk."""
    img_list, student_ids = load_student_images(folder_path)
    logger.info("Loaded %d student images.", len(img_list))

    if upload_to_storage:
        init_firebase()
        upload_images_to_storage(folder_path, os.listdir(folder_path))

    logger.info("Encoding started...")
    encode_list_known = find_encodings(img_list)
    encode_list_known_with_ids = [encode_list_known, student_ids]
    logger.info("Encoding complete.")

    with open(output_path, "wb") as f:
        pickle.dump(encode_list_known_with_ids, f)
    logger.info("Encode file saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_encode_file()
