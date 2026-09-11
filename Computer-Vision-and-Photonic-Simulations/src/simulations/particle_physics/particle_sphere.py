#!/usr/bin/env python3
"""
3D Gesture-Controlled Particle Sphere
======================================

A single-file Python app: 3000 particles arranged on a sphere, rendered with a
simple hand-rolled 3D projection (no game engine / 3D framework), controlled
live via webcam + MediaPipe Hands.

Install:
    pip install opencv-python mediapipe pygame numpy

Run:
    python particle_sphere.py

Controls:
    - One hand visible        -> moves the sphere around the screen and spins it
    - Pinch (thumb + index)   -> hold to charge (ring fills up), release to explode
                                  the sphere across the screen, then it reforms
    - Two hands               -> spread them apart to grow the sphere,
                                  bring them together to shrink it
    - Finger count (0-5)      -> changes the particle color
    - "WEBCAM BG" button      -> toggle the live webcam feed as the background
    - ESC / close window      -> quit

Note on performance: 3000 pygame circle draws per frame is a deliberate,
readable, dependency-light choice. If it feels slow on your machine, lower
NUM_PARTICLES below.
"""

import math
import time
import numpy as np
import cv2
import mediapipe as mp
import pygame

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

WIDTH, HEIGHT = 1280, 800
NUM_PARTICLES = 3000
FOV = 900.0
BASE_RADIUS = 190.0
MIN_RADIUS, MAX_RADIUS = 90.0, 320.0

CAM_WIDTH, CAM_HEIGHT = 640, 480

BG_TOP = np.array([14, 15, 22])
BG_BOTTOM = np.array([4, 4, 7])

COLOR_PALETTE = {
    0: (90, 150, 255),   # fist       -> blue
    1: (70, 220, 210),   # 1 finger   -> teal
    2: (110, 230, 120),  # 2 fingers  -> green
    3: (235, 210, 70),   # 3 fingers  -> yellow
    4: (250, 150, 60),   # 4 fingers  -> orange
    5: (235, 90, 190),   # 5 fingers  -> pink
}

PINCH_THRESHOLD = 0.05  # normalized distance between thumb tip & index tip


# --------------------------------------------------------------------------- #
# Small math helpers
# --------------------------------------------------------------------------- #

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-6] = 1e-6
    return v / n


def fibonacci_sphere(n):
    """Evenly-ish distributed points on a unit sphere."""
    i = np.arange(n)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (i / float(n - 1)) * 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = golden_angle * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.stack([x, y, z], axis=1)


def rotate_points(points, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x1 = x * cy - z * sy
    z1 = x * sy + z * cy
    y2 = y * cp - z1 * sp
    z2 = y * sp + z1 * cp
    return np.stack([x1, y2, z2], axis=1)


def project(world_pts, center):
    z = world_pts[:, 2]
    denom = np.maximum(FOV + z, 60.0)
    scale = np.clip(FOV / denom, 0.05, 3.0)
    screen_x = center[0] + world_pts[:, 0] * scale
    screen_y = center[1] + world_pts[:, 1] * scale
    return screen_x, screen_y, scale, z


def count_fingers(lm_xy):
    """lm_xy: (21,2) normalized landmark coords. Generic, mirror-agnostic heuristic."""
    c = 0
    # Thumb: extended if tip is farther from pinky-mcp than the ip joint is.
    if dist2(lm_xy[4], lm_xy[17]) > dist2(lm_xy[3], lm_xy[17]):
        c += 1
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if lm_xy[tip][1] < lm_xy[pip][1] - 0.02:
            c += 1
    return c


# --------------------------------------------------------------------------- #
# Hand tracking
# --------------------------------------------------------------------------- #

class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.65, tracking_conf=0.55):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame_rgb):
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float64,
                )
                hands_data.append(pts)
        return hands_data

    def close(self):
        self.hands.close()


# --------------------------------------------------------------------------- #
# Particle sphere
# --------------------------------------------------------------------------- #

class ParticleSphere:
    def __init__(self, n=NUM_PARTICLES):
        self.n = n
        self.unit = fibonacci_sphere(n)
        self.dir_mix = self.unit.copy()
        self.explode_dirs = self.unit.copy()
        self.radius_mult = np.ones(n)
        self.explode_target_mult = np.ones(n)

        self.radius = BASE_RADIUS
        self.target_radius = BASE_RADIUS

        self.center = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        self.target_center = self.center.copy()

        self.yaw = 0.0
        self.pitch = 0.0
        self.auto_spin = 0.15  # idle spin, rad/sec

        self.color = np.array(COLOR_PALETTE[0], dtype=float)

        self.phase = "idle"  # idle -> charging -> exploding -> holding -> reforming -> idle
        self.phase_t = 0.0
        self.charge = 0.0

        self.CHARGE_TIME = 1.1
        self.EXPLODE_TIME = 0.55
        self.HOLD_TIME = 0.35
        self.REFORM_TIME = 0.9

    def trigger_explosion(self, power):
        noise = np.random.uniform(-1.4, 1.4, size=self.unit.shape)
        self.explode_dirs = normalize_rows(self.unit + noise)
        lo = 2.5 + power * 1.5
        hi = 4.5 + power * 5.5
        self.explode_target_mult = np.random.uniform(lo, hi, size=self.n)
        self.phase = "exploding"
        self.phase_t = 0.0

    def update(self, dt, hands_count, pinch_active, spread_target,
               move_target, spin_input, finger_color, screen_center):
        # color smoothing
        self.color = lerp(self.color, finger_color.astype(float), min(1.0, dt * 6))

        # position smoothing
        self.target_center = move_target if move_target is not None else screen_center
        self.center = lerp(self.center, self.target_center, min(1.0, dt * 8))

        # size smoothing
        if spread_target is not None:
            self.target_radius = clamp(spread_target, MIN_RADIUS, MAX_RADIUS)
        self.radius = lerp(self.radius, self.target_radius, min(1.0, dt * 6))

        # rotation
        self.yaw += self.auto_spin * dt + spin_input[0]
        self.pitch += spin_input[1]
        self.pitch = clamp(self.pitch, -1.1, 1.1)

        # charge / explode / reform state machine
        if self.phase == "idle":
            self.radius_mult = np.ones(self.n)
            self.dir_mix = self.unit
            if pinch_active and hands_count == 1:
                self.phase = "charging"
                self.charge = 0.0

        elif self.phase == "charging":
            if pinch_active and hands_count == 1:
                self.charge = min(1.0, self.charge + dt / self.CHARGE_TIME)
                pulse = 1.0 + 0.04 * self.charge * math.sin(time.time() * 18)
                self.radius_mult = np.full(self.n, pulse)
            else:
                power = self.charge
                self.charge = 0.0
                self.trigger_explosion(power)

        elif self.phase == "exploding":
            self.phase_t += dt
            t = clamp(self.phase_t / self.EXPLODE_TIME, 0.0, 1.0)
            te = ease_out_cubic(t)
            self.radius_mult = lerp(1.0, self.explode_target_mult, te)
            self.dir_mix = normalize_rows(lerp(self.unit, self.explode_dirs, te))
            if t >= 1.0:
                self.phase = "holding"
                self.phase_t = 0.0

        elif self.phase == "holding":
            self.phase_t += dt
            if self.phase_t >= self.HOLD_TIME:
                self.phase = "reforming"
                self.phase_t = 0.0

        elif self.phase == "reforming":
            self.phase_t += dt
            t = clamp(self.phase_t / self.REFORM_TIME, 0.0, 1.0)
            tr = ease_in_out(t)
            self.radius_mult = lerp(self.explode_target_mult, 1.0, tr)
            self.dir_mix = normalize_rows(lerp(self.explode_dirs, self.unit, tr))
            if t >= 1.0:
                self.phase = "idle"
                self.phase_t = 0.0
                self.radius_mult = np.ones(self.n)
                self.dir_mix = self.unit.copy()

    def get_world_points(self):
        rotated = rotate_points(self.dir_mix, self.yaw, self.pitch)
        world = rotated * (self.radius_mult[:, None]) * self.radius
        return world


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #

def make_bg_gradient(w, h):
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / h
        color = (BG_TOP * (1 - t) + BG_BOTTOM * t).astype(int)
        pygame.draw.line(surf, tuple(color), (0, y), (w, y))
    return surf


def draw_text_shadow(surface, font, text, pos, color, shadow=(0, 0, 0)):
    surface.blit(font.render(text, True, shadow), (pos[0] + 2, pos[1] + 2))
    surface.blit(font.render(text, True, color), pos)


def load_font(names, size, bold=False):
    for name in names:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Particle Sphere")
    clock = pygame.time.Clock()

    title_font = load_font(["Georgia", "Times New Roman", "serif"], 34)
    subtitle_font = load_font(["Georgia", "Times New Roman", "serif"], 15)
    ui_font = load_font(["Arial", "Helvetica", "sans-serif"], 16)
    button_font = load_font(["Arial", "Helvetica", "sans-serif"], 15, bold=True)

    bg_gradient = make_bg_gradient(WIDTH, HEIGHT)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    webcam_ok = cap.isOpened()

    tracker = HandTracker()
    sphere = ParticleSphere()

    webcam_bg = False
    button_rect = pygame.Rect(WIDTH - 210, 22, 190, 42)

    prev_primary_pos = None
    fingers = 0
    hands_count = 0

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    webcam_bg = not webcam_bg

        frame_rgb = None
        hands_data = []
        if webcam_ok:
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)  # mirror, selfie-view
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands_data = tracker.process(frame_rgb)

        hands_count = len(hands_data)
        move_target = None
        spin_input = (0.0, 0.0)
        spread_target = None
        pinch_active = False
        finger_color = np.array(COLOR_PALETTE[0], dtype=float)

        if hands_count >= 1:
            primary = hands_data[0]
            palm_xy = primary[9][:2]  # middle finger MCP ~ palm center
            screen_pos = np.array([palm_xy[0] * WIDTH, palm_xy[1] * HEIGHT])
            move_target = screen_pos

            if prev_primary_pos is not None and hands_count == 1:
                dx = (screen_pos[0] - prev_primary_pos[0]) / WIDTH
                dy = (screen_pos[1] - prev_primary_pos[1]) / HEIGHT
                spin_input = (dx * 2.4, dy * 2.4)
            prev_primary_pos = screen_pos

            fingers = count_fingers(primary[:, :2])
            finger_color = np.array(
                COLOR_PALETTE.get(min(fingers, 5), COLOR_PALETTE[0]), dtype=float
            )

            pinch_d = dist2(primary[4][:2], primary[8][:2])
            pinch_active = pinch_d < PINCH_THRESHOLD
        else:
            prev_primary_pos = None
            fingers = 0

        if hands_count >= 2:
            a = hands_data[0][9][:2]
            b = hands_data[1][9][:2]
            d_px = dist2(a, b) * WIDTH
            spread_target = float(np.interp(d_px, [80, 500], [MIN_RADIUS, MAX_RADIUS]))
            mid = (hands_data[0][9][:2] + hands_data[1][9][:2]) / 2.0
            move_target = np.array([mid[0] * WIDTH, mid[1] * HEIGHT])
            spin_input = (0.0, 0.0)
            pinch_active = False

        screen_center = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        sphere.update(dt, hands_count, pinch_active, spread_target,
                      move_target, spin_input, finger_color, screen_center)

        # --------------------------- render --------------------------- #
        if webcam_bg and frame_rgb is not None:
            disp = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
            surf = pygame.surfarray.make_surface(disp.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((5, 6, 10, 110))
            screen.blit(overlay, (0, 0))
        else:
            screen.blit(bg_gradient, (0, 0))

        world_pts = sphere.get_world_points()
        sx, sy, scale, z = project(world_pts, sphere.center)
        order = np.argsort(z)

        col = sphere.color
        for idx in order:
            s = scale[idx]
            radius_px = max(1, int(2.6 * s))
            bright = clamp(0.55 + 0.5 * ((z[idx] + 400.0) / 800.0), 0.35, 1.4)
            c = (
                int(clamp(col[0] * bright, 0, 255)),
                int(clamp(col[1] * bright, 0, 255)),
                int(clamp(col[2] * bright, 0, 255)),
            )
            px, py = int(sx[idx]), int(sy[idx])
            if -50 <= px <= WIDTH + 50 and -50 <= py <= HEIGHT + 50:
                pygame.draw.circle(screen, c, (px, py), radius_px)

        # charge ring
        if sphere.phase == "charging" and sphere.charge > 0.01:
            ring_r = 74
            rect = pygame.Rect(0, 0, ring_r * 2, ring_r * 2)
            rect.center = (int(sphere.center[0]), int(sphere.center[1]))
            pygame.draw.circle(screen, (90, 95, 110), rect.center, ring_r, 2)
            end_angle = -math.pi / 2 + sphere.charge * 2 * math.pi
            pygame.draw.arc(screen, (245, 245, 250), rect, -math.pi / 2, end_angle, 5)

        # title (serif, top-left)
        draw_text_shadow(screen, title_font, "PARTICLE SPHERE", (34, 24), (235, 235, 240))
        draw_text_shadow(screen, subtitle_font, "gesture-controlled particle field",
                          (37, 64), (140, 145, 160))

        # status (bottom-left)
        mode_label = sphere.phase.upper()
        if sphere.phase == "charging":
            mode_label += f"  {int(sphere.charge * 100)}%"
        status_lines = [
            f"HANDS:  {hands_count}",
            f"FINGERS: {fingers if hands_count >= 1 else '-'}",
            f"MODE:    {mode_label}",
        ]
        y0 = HEIGHT - 92
        for i, line in enumerate(status_lines):
            draw_text_shadow(screen, ui_font, line, (34, y0 + i * 24), (170, 175, 190))

        if not webcam_ok:
            draw_text_shadow(screen, ui_font, "No webcam detected", (34, HEIGHT - 116),
                              (220, 120, 120))

        # button (top-right)
        btn_bg = (235, 235, 240) if webcam_bg else (26, 28, 36)
        btn_border = (235, 235, 240) if webcam_bg else (90, 95, 110)
        btn_text_color = (10, 10, 14) if webcam_bg else (200, 205, 215)
        pygame.draw.rect(screen, btn_bg, button_rect, border_radius=10)
        pygame.draw.rect(screen, btn_border, button_rect, width=1, border_radius=10)
        label = "WEBCAM BG: ON" if webcam_bg else "WEBCAM BG: OFF"
        label_surf = button_font.render(label, True, btn_text_color)
        screen.blit(label_surf, label_surf.get_rect(center=button_rect.center))

        pygame.display.flip()

    if webcam_ok:
        cap.release()
    tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
