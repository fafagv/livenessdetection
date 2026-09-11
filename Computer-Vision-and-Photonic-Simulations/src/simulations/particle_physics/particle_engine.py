"""
particle_engine.py — Cinematic "kefir splash" particle swarm engine.

Hand-agnostic: it only consumes a small HandState dict each frame, so the
exact same engine powers both the live webcam version (hand_particle_storm.py)
and the offline synthetic preview (demo_preview.py).

Vectorized with numpy — all 3000 particles are updated in array ops, not a
Python-level per-particle loop, so the physics step stays cheap even at 30fps.
"""

import numpy as np
import cv2
import math
import time


class HandState:
    """One frame's worth of hand info the engine reacts to."""
    __slots__ = ("present", "palm", "palm_vel", "fingertips", "openness",
                 "openness_vel", "live_conf")

    def __init__(self, present=False, palm=(0, 0), palm_vel=(0, 0),
                 fingertips=None, openness=0.0, openness_vel=0.0, live_conf=0.0):
        self.present = present
        self.palm = np.array(palm, dtype=np.float32)
        self.palm_vel = np.array(palm_vel, dtype=np.float32)
        self.fingertips = fingertips if fingertips is not None else []
        self.openness = openness          # 0 (fist) .. 1 (fully spread)
        self.openness_vel = openness_vel  # rate of change -> splash trigger
        self.live_conf = live_conf        # 0..1 heuristic liveness confidence


class ParticleStorm:
    def __init__(self, width, height, n_particles=3000, seed=7):
        self.w, self.h = width, height
        self.n = n_particles
        rng = np.random.default_rng(seed)

        self.pos = rng.uniform([0, 0], [width, height], size=(self.n, 2)).astype(np.float32)
        self.vel = rng.normal(0, 0.3, size=(self.n, 2)).astype(np.float32)
        self.max_life = rng.uniform(60, 220, size=self.n).astype(np.float32)
        self.life = rng.uniform(0, 1, size=self.n).astype(np.float32) * self.max_life
        self.size = rng.uniform(1.0, 3.2, size=self.n).astype(np.float32)
        self.phase = rng.uniform(0, 2 * math.pi, size=self.n).astype(np.float32)
        self.hero = rng.uniform(0, 1, size=self.n) > 0.92   # ~8% are big bright "hero" droplets
        self.size[self.hero] *= 2.2

        self.canvas = np.zeros((height, width, 3), dtype=np.float32)
        self.t = 0

        # tunables — nudge these for a different "commercial" feel
        self.attract_strength = 0.22
        self.swirl_strength = 0.55
        self.splash_power = 14.0
        self.splash_radius = 260.0
        self.splash_thresh = 0.015
        self.drag = 0.03
        self.trail_decay = 0.86
        self.bg_top = np.array([28, 14, 6], dtype=np.float32)     # deep cool navy (BGR)
        self.bg_bottom = np.array([70, 45, 20], dtype=np.float32)  # warm dark amber (BGR)
        self._make_vignette()
        self._make_gradient_bg()

    # ---------------------------------------------------------------- setup
    def _make_gradient_bg(self):
        ramp = np.linspace(0, 1, self.h, dtype=np.float32)[:, None, None]
        grad = self.bg_top[None, None, :] * (1 - ramp) + self.bg_bottom[None, None, :] * ramp
        self.bg = np.repeat(grad, self.w, axis=1)

    def _make_vignette(self):
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cx, cy = self.w / 2, self.h / 2
        d = np.sqrt(((xx - cx) / (self.w / 2)) ** 2 + ((yy - cy) / (self.h / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * d, 0.35, 1.15)[:, :, None]

    # -------------------------------------------------------------- physics
    def step(self, hand: HandState, dt=1.0):
        self.t += dt
        n = self.n

        if hand.present:
            to_palm = hand.palm[None, :] - self.pos
            dist = np.linalg.norm(to_palm, axis=1) + 1e-4
            direction = to_palm / dist[:, None]
            perp = np.stack([-direction[:, 1], direction[:, 0]], axis=1)

            falloff = np.clip(dist / 420.0, 0.0, 1.0)[:, None]
            attract = direction * self.attract_strength * falloff
            # faster hand movement -> stronger swirl, gives the "storm reacting to you" feel
            hand_speed = float(np.linalg.norm(hand.palm_vel))
            swirl = perp * self.swirl_strength * (1.0 / (1.0 + dist[:, None] * 0.008)) \
                    * (0.6 + min(hand_speed / 8.0, 2.5))

            self.vel += (attract + swirl) * dt

            # open-hand burst: splash outward like liquid breaking on impact
            if hand.openness_vel > self.splash_thresh:
                mask = dist < self.splash_radius
                if mask.any():
                    burst = direction[mask] * self.splash_power * hand.openness_vel * 40.0
                    self.vel[mask] += burst.astype(np.float32)

            # fingertip sparkle-pull: hero particles get drawn toward fingertips
            if hand.fingertips:
                tips = np.array(hand.fingertips, dtype=np.float32)
                idx = np.where(self.hero)[0]
                if len(idx):
                    sub = self.pos[idx]
                    d2 = ((sub[:, None, :] - tips[None, :, :]) ** 2).sum(-1)
                    nearest = tips[d2.argmin(axis=1)]
                    pull = (nearest - sub)
                    pd = np.linalg.norm(pull, axis=1, keepdims=True) + 1e-4
                    self.vel[idx] += (pull / pd) * 0.12 * dt
        else:
            # idle ambient drift — gentle turbulent field so it never looks static
            wx = np.sin(self.pos[:, 1] * 0.006 + self.t * 0.02 + self.phase) * 0.06
            wy = np.cos(self.pos[:, 0] * 0.006 + self.t * 0.017 + self.phase) * 0.04
            self.vel += np.stack([wx, wy], axis=1) * dt

        # drag + integrate
        self.vel *= (1.0 - self.drag)
        self.pos += self.vel * dt

        # life cycle
        self.life -= dt
        dead = self.life <= 0
        wrap = (self.pos[:, 0] < -20) | (self.pos[:, 0] > self.w + 20) | \
               (self.pos[:, 1] < -20) | (self.pos[:, 1] > self.h + 20)
        respawn = dead | wrap
        n_respawn = int(respawn.sum())
        if n_respawn:
            rng = np.random.default_rng(int(self.t) + 1)
            if hand.present:
                # respawn drifting outward from the palm, like continuous mist
                ang = rng.uniform(0, 2 * math.pi, n_respawn)
                r = rng.uniform(0, 40, n_respawn)
                self.pos[respawn, 0] = hand.palm[0] + np.cos(ang) * r
                self.pos[respawn, 1] = hand.palm[1] + np.sin(ang) * r
                self.vel[respawn] = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 0.6
            else:
                self.pos[respawn] = rng.uniform([0, 0], [self.w, self.h], (n_respawn, 2))
                self.vel[respawn] = rng.normal(0, 0.2, (n_respawn, 2))
            self.max_life[respawn] = rng.uniform(60, 220, n_respawn)
            self.life[respawn] = self.max_life[respawn]

    # --------------------------------------------------------------- render
    def render(self, hand: HandState, live_label=None):
        # background gradient + fading trail of previous particles
        self.canvas *= self.trail_decay
        self.canvas = np.maximum(self.canvas, self.bg * (1 - self.trail_decay))

        glow = np.zeros_like(self.canvas)
        life_frac = np.clip(self.life / np.maximum(self.max_life, 1e-3), 0, 1)

        for i in range(self.n):
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            if x < 0 or y < 0 or x >= self.w or y >= self.h:
                continue
            lf = life_frac[i]
            base = 235 if self.hero[i] else 200
            # creamy kefir-white with a faint cool-to-warm shimmer
            shimmer = 15 * math.sin(self.phase[i] + self.t * 0.05)
            color = (base + shimmer, base - 5 + shimmer * 0.5, base - 25)
            color = tuple(float(max(0, min(255, c))) * float(lf) for c in color)
            r = max(1, int(self.size[i]))
            cv2.circle(self.canvas, (x, y), r, color, -1, lineType=cv2.LINE_AA)
            if self.hero[i]:
                cv2.circle(glow, (x, y), r * 5, color, -1, lineType=cv2.LINE_AA)

        if glow.any():
            glow = cv2.GaussianBlur(glow, (0, 0), 9)
            self.canvas = cv2.addWeighted(self.canvas, 1.0, glow, 0.35, 0)

        frame = np.clip(self.canvas * self.vignette, 0, 255).astype(np.uint8)

        if hand.present:
            cv2.circle(frame, tuple(hand.palm.astype(int)), 6, (255, 255, 255), 2, cv2.LINE_AA)

        if live_label:
            color = (90, 230, 90) if live_label.startswith("LIVE") else (60, 60, 230)
            cv2.putText(frame, live_label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, color, 2, cv2.LINE_AA)

        return frame
