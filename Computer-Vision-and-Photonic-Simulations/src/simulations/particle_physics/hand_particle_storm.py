"""
hand_particle_storm.py — Run this LOCALLY (needs a webcam + display).

Controls a 3000-particle cinematic "kefir splash" storm with your hand:
  - Move your palm  -> particles swirl and follow it
  - Move it fast     -> swirl intensifies
  - Snap hand open   -> outward splash burst
  - Hold steady, open -> hero droplets orbit your fingertips

Also runs a lightweight hand-liveness heuristic: real hands have constant
micro-jitter (tremor, blood-flow-driven micro-motion); a photo/screen held up
to the camera is nearly perfectly rigid. We track the variance of the hand's
internal shape (fingertip-to-wrist distances) over a short rolling window —
low variance sustained over time flags "STATIC" (possible spoof).

  NOTE: this is a simple anti-spoof heuristic, not biometric-grade liveness
  security — don't use it to gate anything sensitive.

Install:
    pip install opencv-python mediapipe numpy

Keys:
    r  — toggle recording to output.mp4
    q  — quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

try:
    from .particle_engine import ParticleStorm, HandState
except ImportError:  # allows `python hand_particle_storm.py` as a standalone script too
    from particle_engine import ParticleStorm, HandState

WIDTH, HEIGHT = 960, 540
N_PARTICLES = 3000

LIVENESS_WINDOW = 20          # frames of history for the jitter check
LIVENESS_STD_THRESH = 0.35    # px; below this sustained -> flagged static
LIVENESS_MIN_FRAMES = 15      # need this many frames of history before judging

FINGERTIP_IDS = [4, 8, 12, 16, 20]
WRIST_ID = 0


def landmark_signature(landmarks, w, h):
    """Fingertip-to-wrist distances — a compact 'shape' fingerprint per frame."""
    wrist = np.array([landmarks[WRIST_ID].x * w, landmarks[WRIST_ID].y * h])
    sig = []
    for tid in FINGERTIP_IDS:
        p = np.array([landmarks[tid].x * w, landmarks[tid].y * h])
        sig.append(np.linalg.norm(p - wrist))
    return np.array(sig, dtype=np.float32)


def openness_from_signature(sig):
    # normalize against the largest observed spread so it behaves ~0..1
    return float(np.clip(sig.mean() / 220.0, 0.0, 1.5))


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)

    storm = ParticleStorm(WIDTH, HEIGHT, n_particles=N_PARTICLES)

    sig_history = deque(maxlen=LIVENESS_WINDOW)
    prev_palm = None
    prev_openness = 0.0

    recording = False
    writer = None
    pTime = time.time()

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = cv2.flip(img, 1)  # mirror, feels natural for hand control
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_state = HandState(present=False)
        live_label = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            h, w = HEIGHT, WIDTH

            palm = np.array([lm[9].x * w, lm[9].y * h], dtype=np.float32)  # middle_mcp ~ palm center
            tips = [(lm[i].x * w, lm[i].y * h) for i in FINGERTIP_IDS]

            sig = landmark_signature(lm, w, h)
            sig_history.append(sig)
            openness = openness_from_signature(sig)

            palm_vel = (palm - prev_palm) if prev_palm is not None else np.zeros(2, np.float32)
            openness_vel = max(openness - prev_openness, 0.0)
            prev_palm, prev_openness = palm, openness

            # --- liveness heuristic ---
            live_conf = 0.5
            if len(sig_history) >= LIVENESS_MIN_FRAMES:
                arr = np.stack(sig_history, axis=0)
                jitter = float(arr.std(axis=0).mean())
                live_conf = float(np.clip(jitter / (LIVENESS_STD_THRESH * 3), 0.0, 1.0))
                live_label = f"LIVE  {live_conf*100:.0f}%" if jitter > LIVENESS_STD_THRESH \
                    else f"STATIC?  jitter={jitter:.2f}px"
            else:
                live_label = "calibrating..."

            hand_state = HandState(present=True, palm=palm, palm_vel=palm_vel,
                                    fingertips=tips, openness=openness,
                                    openness_vel=openness_vel, live_conf=live_conf)
        else:
            prev_palm = None
            prev_openness = 0.0
            sig_history.clear()

        storm.step(hand_state)
        frame = storm.render(hand_state, live_label=live_label)

        cTime = time.time()
        fps = 1.0 / max(cTime - pTime, 1e-6)
        pTime = cTime
        cv2.putText(frame, f"{fps:.0f} fps", (WIDTH - 110, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "[r] record  [q] quit", (24, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        if recording:
            cv2.circle(frame, (WIDTH - 30, 20), 8, (0, 0, 255), -1)
            if writer is not None:
                writer.write(frame)

        cv2.imshow("Particle Storm — Hand Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                          30, (WIDTH, HEIGHT))
                print("Recording started -> output.mp4")
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped.")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
