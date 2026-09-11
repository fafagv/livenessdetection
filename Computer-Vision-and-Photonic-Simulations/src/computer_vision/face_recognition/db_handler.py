"""
db_handler.py — Firebase Realtime Database interface for the face-attendance system.

Refactored from the original AddDatatoDatabase.py. Credentials are now loaded
from environment variables / a local service-account file instead of being
hardcoded, and the module exposes reusable functions instead of running as a
one-off script.

Environment variables (see .env.example at the repo root):
    FIREBASE_SERVICE_ACCOUNT_PATH   Path to your serviceAccountKey.json
    FIREBASE_DATABASE_URL           Realtime Database URL
    FIREBASE_STORAGE_BUCKET         (optional) Storage bucket name
"""

import os
import logging
from typing import Optional

import firebase_admin
from firebase_admin import credentials, db, storage

logger = logging.getLogger(__name__)

_app_initialized = False


def init_firebase(
    service_account_path: Optional[str] = None,
    database_url: Optional[str] = None,
    storage_bucket: Optional[str] = None,
) -> None:
    """Initialize the Firebase app exactly once, pulling defaults from env vars."""
    global _app_initialized
    if _app_initialized:
        return

    service_account_path = service_account_path or os.environ.get(
        "FIREBASE_SERVICE_ACCOUNT_PATH", "config/serviceAccountKey.json"
    )
    database_url = database_url or os.environ.get("FIREBASE_DATABASE_URL", "")
    storage_bucket = storage_bucket or os.environ.get("FIREBASE_STORAGE_BUCKET", "")

    if not os.path.exists(service_account_path):
        raise FileNotFoundError(
            f"Firebase service account file not found at '{service_account_path}'. "
            "Copy config/serviceAccountKey.json.example to config/serviceAccountKey.json "
            "and fill in your real credentials (never commit the real file)."
        )

    cred = credentials.Certificate(service_account_path)
    options = {"databaseURL": database_url}
    if storage_bucket:
        options["storageBucket"] = storage_bucket

    firebase_admin.initialize_app(cred, options)
    _app_initialized = True
    logger.info("Firebase app initialized.")


def add_student(student_id: str, data: dict) -> None:
    """Write a single student's record to /Students/{student_id}."""
    init_firebase()
    ref = db.reference(f"Students/{student_id}")
    ref.set(data)
    logger.info("Added/updated student record for id=%s", student_id)


def get_student(student_id: str) -> Optional[dict]:
    """Fetch a single student's record."""
    init_firebase()
    ref = db.reference(f"Students/{student_id}")
    return ref.get()


def update_attendance(student_id: str, total_attendance: int, timestamp: str) -> None:
    """Update a student's attendance counter and last-seen timestamp."""
    init_firebase()
    ref = db.reference(f"Students/{student_id}")
    ref.child("total_attendance").set(total_attendance)
    ref.child("last_attendance_time").set(timestamp)


def get_storage_bucket():
    """Return the initialized Firebase Storage bucket handle."""
    init_firebase()
    return storage.bucket()


if __name__ == "__main__":
    # Example / smoke-test usage (mirrors the original AddDatatoDatabase.py seed data).
    logging.basicConfig(level=logging.INFO)
    init_firebase()
    add_student(
        "321654",
        {
            "name": "Murtaza Hassan",
            "major": "Robotics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34",
        },
    )
