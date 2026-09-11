"""
face_attendance_app.py — Main application entry point for face-recognition
attendance logging.

Refactored from main2.py. The original mixed Firebase init, camera capture,
face matching, and Tkinter-less UI rendering into one 150-line top-level
script with no functions and no exception handling. This version delegates
to:
    - src/computer_vision/face_recognition/db_handler.py  (Firebase I/O)
    - src/computer_vision/face_recognition/tracker.py      (detect + match)

and wraps the capture loop in proper setup/teardown.

Run:
    python -m apps.face_attendance_app
"""

import logging
import os
import pickle
from datetime import datetime

import cv2
import cvzone
import numpy as np

from src.computer_vision.face_recognition import db_handler
from src.computer_vision.face_recognition.tracker import FaceTracker

logger = logging.getLogger(__name__)


def load_mode_images(folder_path: str):
    if not os.path.isdir(folder_path):
        logger.warning("Mode images folder '%s' not found.", folder_path)
        return []
    return [cv2.imread(os.path.join(folder_path, p)) for p in sorted(os.listdir(folder_path))]


def run(
    camera_index: int = 0,
    background_path: str = "data/resources/attendance_background.png",
    mode_images_folder: str = "data/resources/attendance_modes",
    encode_file_path: str = "EncodeFile.p",
):
    db_handler.init_firebase()
    bucket = db_handler.get_storage_bucket()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, 640)
    cap.set(4, 480)

    img_background = cv2.imread(background_path)
    if img_background is None:
        logger.error("Background image '%s' not found — cannot render the attendance UI.", background_path)
        return

    img_mode_list = load_mode_images(mode_images_folder)
    if not img_mode_list:
        logger.error("No mode images found in '%s'.", mode_images_folder)
        return

    tracker = FaceTracker(encode_file_path)

    mode_type = 0
    counter = 0
    student_id = -1
    img_student = None
    student_info = None

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            matches = tracker.detect_and_match(img_small_rgb)

            img_background[162:162 + 480, 55:55 + 640] = img
            img_background[44:44 + 633, 808:808 + 414] = img_mode_list[mode_type]

            if matches:
                for matched_id, face_loc, _distance in matches:
                    if matched_id is None:
                        continue
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    img_background = cvzone.cornerRect(img_background, bbox, rt=0)
                    student_id = matched_id

                    if counter == 0:
                        cvzone.putTextRect(img_background, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", img_background)
                        cv2.waitKey(1)
                        counter = 1
                        mode_type = 1

                if counter != 0:
                    if counter == 1:
                        student_info = db_handler.get_student(student_id)
                        if student_info is None:
                            logger.warning("No DB record found for student id=%s", student_id)
                            counter = 0
                        else:
                            try:
                                blob = bucket.get_blob(f"Images/{student_id}.png")
                                array = np.frombuffer(blob.download_as_string(), np.uint8)
                                img_student = cv2.imdecode(array, cv2.IMREAD_COLOR)
                            except Exception:
                                logger.exception("Failed to fetch student photo from storage.")
                                img_student = None

                            datetime_object = datetime.strptime(
                                student_info["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
                            )
                            seconds_elapsed = (datetime.now() - datetime_object).total_seconds()

                            if seconds_elapsed > 30:
                                new_total = student_info["total_attendance"] + 1
                                db_handler.update_attendance(
                                    student_id, new_total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                )
                                student_info["total_attendance"] = new_total
                            else:
                                mode_type = 3
                                counter = 0
                                img_background[44:44 + 633, 808:808 + 414] = img_mode_list[mode_type]

                    if mode_type != 3 and student_info is not None:
                        if 10 < counter < 20:
                            mode_type = 2
                        img_background[44:44 + 633, 808:808 + 414] = img_mode_list[mode_type]

                        if counter <= 10:
                            cv2.putText(img_background, str(student_info["total_attendance"]), (861, 125),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                            cv2.putText(img_background, str(student_info["major"]), (1006, 550),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(img_background, str(student_id), (1006, 493),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(img_background, str(student_info["standing"]), (910, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(img_background, str(student_info["year"]), (1025, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(img_background, str(student_info["starting_year"]), (1125, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                            (w, h), _ = cv2.getTextSize(student_info["name"], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                            offset = (414 - w) // 2
                            cv2.putText(img_background, str(student_info["name"]), (808 + offset, 445),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                            if img_student is not None:
                                img_background[175:175 + 216, 909:909 + 216] = img_student

                        counter += 1
                        if counter >= 20:
                            counter = 0
                            mode_type = 0
                            student_info = None
                            img_student = None
                            img_background[44:44 + 633, 808:808 + 414] = img_mode_list[mode_type]
            else:
                mode_type = 0
                counter = 0

            cv2.imshow("Face Attendance", img_background)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
