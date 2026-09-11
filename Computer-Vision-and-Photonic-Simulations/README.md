# Computer Vision & Photonic Simulations

A unified repository combining two sub-systems built during a series of
computer-vision experiments and a photonic-crystal waveguide research
project:

1. **Computer Vision** — MediaPipe/OpenCV-based face recognition attendance,
   hand-gesture control (mouse, volume, zoom, slide presentations, games),
   and MediaPipe-based sign-language data collection/testing.
2. **Photonic Simulations** — Photonic-crystal slab waveguide mode analysis,
   analyte sensitivity, and wavelength-shift extraction, plus a couple of
   standalone particle-physics visual simulations and a PyTorch LSTM
   time-series demo.

## Repository layout

```
Computer-Vision-and-Photonic-Simulations/
├── config/                      # Firebase config template (no real secrets)
├── data/                        # Datasets, presentation slides, UI assets (gitignored except .gitkeep)
├── src/
│   ├── computer_vision/
│   │   ├── face_recognition/    # Firebase-backed face encoding, tracking, DB I/O
│   │   ├── gesture_control/     # Hand tracking, gesture engine, mouse/volume/zoom/finger-count
│   │   └── sign_language/       # Landmark data collection + Keras model testing
│   ├── simulations/
│   │   ├── particle_physics/    # 3D/2D webcam-driven particle simulations
│   │   └── photonics/           # Mode properties, sensitivity, wavelength-shift analysis
│   └── deep_learning/
│       └── lstm_time_series/    # PyTorch LSTM sine-wave forecaster
├── apps/                        # Runnable entry points, one per end-user application
├── legacy_misc/                 # Out-of-scope scripts kept for reference only (see its README)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <this-repo-url>
cd Computer-Vision-and-Photonic-Simulations
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`face_recognition` depends on `dlib`, which needs CMake and a C++ compiler
to build from source on most platforms — install those first if `pip
install` fails on that package. `pycaw`/`comtypes` (used only by
`volume_control.py`) are Windows-only and are skipped automatically on
other platforms.

### Firebase setup (face recognition only)

1. Copy `config/serviceAccountKey.json.example` to
   `config/serviceAccountKey.json` and fill in your real Firebase
   service-account credentials — **never commit the real file** (it's
   gitignored).
2. Copy `.env.example` to `.env` and fill in your `FIREBASE_DATABASE_URL`
   and `FIREBASE_STORAGE_BUCKET`.
3. Load the `.env` file into your shell (e.g. via `python-dotenv` or
   `export $(cat .env | xargs)`) before running `face_attendance_app.py`.

## Usage

Run any application as a module from the repository root:

| App | Command | Description |
|---|---|---|
| Face Attendance | `python -m apps.face_attendance_app` | Recognizes enrolled students via webcam and logs attendance to Firebase. Run `src/computer_vision/face_recognition/encoder.py` first to build `EncodeFile.p` from `data/images/`. |
| Gesture Presentation | `python -m apps.gesture_presentation_app` | Controls a slide deck (`data/presentation_slides/`) with hand gestures: swipe, point, draw, undo. |
| Gesture Games | `python -m apps.gesture_games_app` | A timed hand-distance "pop the target" game with on-screen cm estimation. |
| Photonic Analyzer | `python -m apps.photonic_analyzer_app --wavelength 1.55 --analyte-n 1.45 --plot` | CLI for mode/sensitivity/wavelength-shift analysis of the photonic-crystal waveguide sensor. |

Standalone simulations (no `apps/` wrapper needed — run directly):

```bash
python -m src.simulations.particle_physics.hand_particle_storm   # webcam hand-controlled particle storm
python -m src.simulations.particle_physics.particle_sphere       # gesture-controlled 3D particle sphere (pygame)
python -m src.deep_learning.lstm_time_series.lstm_predictor       # trains & plots the sine-wave LSTM forecaster
```

Gesture-control building blocks (`hand_tracker.py`, `gesture_detector.py`,
`finger_counter.py`, `volume_control.py`, `virtual_mouse.py`,
`virtual_zoom.py`) can each also be run standalone as a demo via
`python -m src.computer_vision.gesture_control.<module_name>`.

Sign-language pipeline:

```bash
python -m src.computer_vision.sign_language.data_collection   # press 's' to save a labeled hand crop
python -m src.computer_vision.sign_language.model_tester       # run a trained Keras model against the webcam
```

## What changed during the refactor

This repository was consolidated from an unstructured collection of ~30
loose scripts. Notable fixes made along the way:

- **Removed duplicate/dead code**: four near-identical copies of the hand
  tracking module were collapsed into `hand_tracker.py`; a byte-identical
  duplicate of `particle_sphere.py` was dropped.
- **Fixed real bugs**, including: a duplicate function definition that
  silently shadowed the real implementation in `mode_properties.py`; a
  `hands = detector.findHands(...)` unpacking bug in the games app that
  would have crashed on the first detected hand; and a badly broken LSTM
  script (`lstm_predictor.py`) with multiple typos, an undefined variable,
  and a genuine `SyntaxError` that meant it never actually ran in its
  original form.
- **Removed hardcoded secrets**: Firebase credentials now load from
  `config/serviceAccountKey.json` (gitignored) and environment variables
  instead of being loaded from a bare `"serviceAccountKey.json"` string
  with an empty database URL baked into the source.
- **Added exception handling and camera-release cleanup** to every
  webcam loop, which the originals largely lacked.
- **Excluded out-of-scope files** (parking-lot detection, a Kinect-based
  sign-language demo, a YOLOv5+GPT-2 hybrid detector, and a standalone
  perceptron demo) into `legacy_misc/`, since they don't belong to either
  the Computer Vision or Photonic Simulations sub-systems.

See `legacy_misc/README.md` for details on the excluded files.
