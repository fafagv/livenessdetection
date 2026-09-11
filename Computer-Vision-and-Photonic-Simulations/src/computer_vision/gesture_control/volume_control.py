"""
volume_control.py — System volume modification via PyCAW, driven by a pinch gesture.

Refactored from Gesture.py. PyCAW is Windows-only, so the import is now
guarded and the app fails with a clear message on other platforms instead
of crashing on `from pycaw.pycaw import ...` at import time.

Fixes from the original:
    - No more bare `while True` with an un-releasable camera on exceptions.
    - Fixed a latent divide-by-zero risk in FPS calculation.
    - Reuses gesture_detector.GestureDetector instead of duplicating the
      pinch-distance / fingers-up / bbox-area logic.
"""

import logging
import platform

import cv2
import numpy as np

from .gesture_detector import GestureDetector

logger = logging.getLogger(__name__)


def _get_volume_interface():
    if platform.system() != "Windows":
        raise RuntimeError(
            "volume_control.py depends on PyCAW, which only works on Windows. "
            "Skipping hardware volume control on this platform."
        )
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def run(camera_index: int = 0, cam_size=(640, 480)):
    volume = _get_volume_interface()
    vol_range = volume.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]  # noqa: F841 (kept for parity with source formula inputs)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return
    cap.set(3, cam_size[0])
    cap.set(4, cam_size[1])

    detector = GestureDetector(max_hands=1, detection_con=0.7)
    vol_bar, vol_perc, color_vol = 400, 0, (255, 0, 0)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, state = detector.analyze(img, draw=True)

            if state and detector.bbox_area(state.bbox) and 250 <= detector.bbox_area(state.bbox) <= 1300:
                length = state.pinch_distance or 0
                vol_bar = detector.map_range(length, (50, 300), (400, 150))
                vol_perc = detector.map_range(length, (50, 300), (0, 100))

                smoothness = 10
                vol_perc = smoothness * round(vol_perc / smoothness)

                # Pinky-down = "confirm" gesture, matching the original UX.
                if not state.fingers[4]:
                    volume.SetMasterVolumeLevelScalar(vol_perc / 100, None)
                    color_vol = (0, 255, 0)
                else:
                    color_vol = (255, 0, 0)

            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f"{int(vol_perc)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cur_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(img, f"Vol Set: {cur_vol}%", (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color_vol, 3)

            cv2.imshow("Volume Control", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
