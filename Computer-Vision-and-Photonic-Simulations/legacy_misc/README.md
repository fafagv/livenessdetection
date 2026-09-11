# legacy_misc/

These 7 scripts were found in the original project workspace but don't belong
to any module in the target architecture (Computer Vision, Photonic
Simulations, or Deep Learning). They're kept here for reference only —
**nothing in `src/` or `apps/` imports from this folder**, and it isn't
covered by `requirements.txt`.

| File | Original name | What it is |
|---|---|---|
| `parking_space_picker.py` | `ParkingSpacePicker.py` | Interactive tool to mark parking-space ROIs on a video frame by mouse click. |
| `parking_space_detector.py` | `main (2).py` | Runs occupied/empty detection on a parking-lot video using the picked ROIs. |
| `parking_space_threshold_tuner.py` | `Trackbars.py` | OpenCV trackbar UI for tuning the parking detector's thresholding parameters. |
| `image_color_grid_analyzer.py` | `main (3).py` | Splits an arbitrary image into a grid and reports the dominant color per cell. |
| `kinect_sign_language_legacy.py` | `Sign.py` | An older sign-language gesture reader built on the Microsoft Kinect (`freenect`) and `vlc`/`playsound` audio playback — unrelated to, and technically incompatible with, the MediaPipe-based sign-language pipeline in `src/computer_vision/sign_language/`. |
| `yolo_gpt2_hybrid_detector.py` | `detection.py` | A YOLOv5 object detector combined with a GPT-2 text model for "sequence analysis" — a different detection stack (not MediaPipe/face_recognition-based) with no other file in this project depending on it. |
| `sigmoid_perceptron_demo.py` | `main1.py` | A standalone single-neuron sigmoid/perceptron demo, unrelated to the LSTM time-series model that the target architecture's `deep_learning/` module is scoped to. |

If any of these are meant to become part of the maintained project, let's
give them a proper home in the architecture (or a follow-up repo) rather
than leaving them here unlinked.
