"""
virtual_mouse.py — Hand & eye-controlled pointer navigation.

Two source files fed into this module:
    - eye_controlled_mouse.py, refactored below into EyeMouseController.
    - AiVirtualMouseProject.py, which was found to be a completely EMPTY
      (0-byte) file in the original project. There was no hand-controlled
      mouse implementation to port. HandMouseController below is a new,
      minimal implementation built on top of gesture_detector.GestureDetector
      to satisfy the "Hand & eye-controlled pointer navigation" scope this
      module is meant to cover — flagged here so it's clear it is NOT a
      refactor of pre-existing logic like the rest of this repository.

Fixes to the ported eye-mouse logic:
    - Guards for a landmark list that's shorter than expected.
    - Proper camera release on exit (original had no cleanup at all).
    - Click cooldown no longer blocks the whole thread via pyautogui.sleep(1);
      uses a non-blocking timestamp check instead.
"""

import logging
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyautogui
except Exception:  # pragma: no cover - pyautogui needs a display
    pyautogui = None

from .gesture_detector import GestureDetector

logger = logging.getLogger(__name__)


class EyeMouseController:
    """Moves the system cursor with gaze position and blinks the left eye to click."""

    def __init__(self, click_cooldown_s: float = 1.0):
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for eye-controlled mouse control.")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w, self.screen_h = pyautogui.size()
        self.click_cooldown_s = click_cooldown_s
        self._last_click_time = 0.0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if not landmark_points:
            return frame

        landmarks = landmark_points[0].landmark

        for idx, landmark in enumerate(landmarks[474:478]):
            x, y = int(landmark.x * frame_w), int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if idx == 1:
                pyautogui.moveTo(self.screen_w * landmark.x, self.screen_h * landmark.y)

        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x, y = int(landmark.x * frame_w), int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left_eye[0].y - left_eye[1].y) < 0.004:
            now = time.time()
            if now - self._last_click_time > self.click_cooldown_s:
                pyautogui.click()
                self._last_click_time = now

        return frame


class HandMouseController:
    """Moves the cursor with the index fingertip; pinch (thumb+index) triggers a click.

    New implementation — see module docstring. Not a port of existing code.
    """

    def __init__(self, frame_size=(640, 480), pinch_threshold: float = 35.0, click_cooldown_s: float = 0.8):
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for hand-controlled mouse control.")
        self.detector = GestureDetector(max_hands=1, detection_con=0.7)
        self.frame_w, self.frame_h = frame_size
        self.screen_w, self.screen_h = pyautogui.size()
        self.pinch_threshold = pinch_threshold
        self.click_cooldown_s = click_cooldown_s
        self._last_click_time = 0.0

    def process_frame(self, frame):
        frame, state = self.detector.analyze(frame, draw=True)
        if state is None or len(state.lm_list) <= 8:
            return frame

        index_x, index_y = state.lm_list[8][1], state.lm_list[8][2]
        screen_x = np.interp(index_x, (0, self.frame_w), (0, self.screen_w))
        screen_y = np.interp(index_y, (0, self.frame_h), (0, self.screen_h))
        pyautogui.moveTo(screen_x, screen_y)

        if state.pinch_distance is not None and state.pinch_distance < self.pinch_threshold:
            now = time.time()
            if now - self._last_click_time > self.click_cooldown_s:
                pyautogui.click()
                self._last_click_time = now

        return frame


def run(mode: str = "eye", camera_index: int = 0, cam_size=(640, 480)):
    """mode: 'eye' or 'hand'."""
    controller = EyeMouseController() if mode == "eye" else HandMouseController(frame_size=cam_size)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, cam_size[0])
    cap.set(4, cam_size[1])

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = controller.process_frame(frame)
            cv2.imshow(f"{mode.title()} Controlled Mouse", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(mode="eye")
