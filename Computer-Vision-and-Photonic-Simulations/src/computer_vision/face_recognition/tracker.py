"""
tracker.py — Real-time face detection & matching logic.

Extracted from the original main2.py, which mixed camera capture, UI
rendering, and face matching in a single monolithic loop. This module
isolates the reusable "detect faces in a frame and match against known
encodings" logic so it can be unit-tested and reused outside the
attendance UI (apps/face_attendance_app.py).
"""

import logging
import pickle
from typing import List, Optional, Tuple

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)


class FaceTracker:
    """Wraps face_recognition detection + matching against a known encoding set."""

    def __init__(self, encode_file_path: str = "EncodeFile.p"):
        self.encode_file_path = encode_file_path
        self.known_encodings: List[np.ndarray] = []
        self.known_ids: List[str] = []
        self._load_encodings()

    def _load_encodings(self) -> None:
        try:
            with open(self.encode_file_path, "rb") as f:
                self.known_encodings, self.known_ids = pickle.load(f)
            logger.info("Loaded %d known face encodings.", len(self.known_encodings))
        except FileNotFoundError:
            logger.warning(
                "Encode file '%s' not found. Run encoder.generate_encode_file() first.",
                self.encode_file_path,
            )
            self.known_encodings, self.known_ids = [], []

    def detect_and_match(
        self, frame_rgb_small: np.ndarray
    ) -> List[Tuple[Optional[str], Tuple[int, int, int, int], float]]:
        """
        Detect faces in a (typically downscaled) RGB frame and match each
        against the known encoding set.

        Returns a list of (student_id_or_None, face_location, distance) tuples.
        face_location is (top, right, bottom, left) as returned by face_recognition.
        """
        face_locations = face_recognition.face_locations(frame_rgb_small)
        face_encodings = face_recognition.face_encodings(frame_rgb_small, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            if not self.known_encodings:
                results.append((None, location, float("inf")))
                continue

            face_distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_match_index = int(np.argmin(face_distances))
            matches = face_recognition.compare_faces(self.known_encodings, encoding)

            if matches[best_match_index]:
                student_id = self.known_ids[best_match_index]
            else:
                student_id = None

            results.append((student_id, location, float(face_distances[best_match_index])))

        return results
