#!/usr/bin/env python3
"""
Head-pose monitor selector (gaze-focus).

Reads your webcam, figures out which way your head is pointed in TWO axes
(left/right turn = "yaw", up/down tilt = "pitch"), and continuously writes the
current monitor "zone" to ~/.gaze/state.json:

    0 = LAPTOP   1 = TOP monitor   2 = RIGHT monitor

This handles non-linear monitor layouts. The default labels assume:

    - LAPTOP: on the desk, you look DOWN at it
    - TOP:    mounted above the laptop, you look UP at it
    - RIGHT:  a monitor off to your right, you TURN right toward it

Because laptop and top are both straight ahead, head turn alone can't tell them
apart -- we need the up/down tilt too.

How it decides: you calibrate once by looking at each monitor and pressing its
number key (1/2/3). That records the head pose for that monitor; each frame is
then classified to whichever calibrated point it's closest to. No hand-tuned
thresholds, no assumptions about geometry. The head pose comes from the model's
real 3D facial-orientation matrix (not 2D landmark ratios) and is run through a
One-Euro filter, so it's smooth when you hold still but snappy when you turn.
Calibration averages ~0.4 s of frames, so hold your gaze on the target for a
beat before pressing the key. A stickiness margin keeps the zone stable so it
doesn't flicker at the boundaries.

EYE GAZE (sub-window selection):
Head pose can pick the monitor, but it can't tell apart two windows split on
the SAME screen (e.g. a top + bottom split on the vertical monitor) -- you move
your eyes for that, not your head. So we also estimate vertical EYE gaze with a
small appearance-based neural net: an ONNX gaze model (trained on Gaze360) run
on a crop of your face. Unlike geometric iris-ratio tricks it stays accurate
even when your head is turned, which is exactly when those tricks fall apart.
Calibrate a screen's split by looking at the top window (press 't') and the
bottom window (press 'b') -- hold your gaze for a beat so it averages cleanly;
from then on we emit a continuous "gaze_y" in [0,1] (0 = top, 1 = bottom)
whenever that screen is the active zone. Hammerspoon uses gaze_y to focus the
exact window at that height. This works for any vertical split (2, 3, ...
windows), not just two.

It NEVER changes window focus itself. It just keeps the current zone (and, when
calibrated, gaze_y) live -- plus, if you pass --auto-focus, a flag in the state
file telling Hammerspoon to commit hands-free. The commit (actually switching
focus) is done by Hammerspoon: when you press your trigger (a key or hotkey), or
-- with --auto-focus -- automatically once you've dwelt on a window for
--auto-focus-seconds (default 3). Eyes point, the trigger (or a steady look) commits.

Live keys in the preview window:
    1  -> ADD a LAPTOP-monitor pose sample (press again from other postures)
    2  -> ADD a TOP-monitor pose sample
    3  -> ADD a RIGHT-monitor pose sample
    u  -> undo the last pose sample you added
    t  -> record TOP window of the current monitor (look at it first)
    b  -> record BOTTOM window of the current monitor (look at it first)
    x  -> clear the current monitor's top/bottom (eye) calibration
    r  -> reset / clear ALL calibration (monitors + eye)
    [  -> stickier (harder to switch zones)
    ]  -> looser (easier to switch zones)
    q / ESC -> quit

Each 1/2/3 press ADDS a sample rather than overwriting, so you can record a
monitor from several head/body postures (leaning back, sitting upright, etc.)
and a frame is matched to the NEAREST sample of each zone -- robust to the way
you actually shift around.
"""

import argparse
import json
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HOME = os.path.expanduser("~")
GAZE_DIR = os.path.join(HOME, ".gaze")
STATE_PATH = os.path.join(GAZE_DIR, "state.json")
CONFIG_PATH = os.path.join(GAZE_DIR, "config.json")
MODEL_PATH = os.path.join(GAZE_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe FaceMesh landmark indices (same in the Tasks API)
NOSE = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263
CHIN = 152

# Eyelid + corner landmarks, used ONLY to detect blinks/squints so we can hold
# the last good gaze read instead of trusting a half-shut eye. The vertical-gaze
# signal itself comes from the ONNX model below, not from these points.
R_EYE_TOP, R_EYE_BOT, R_EYE_IN, R_EYE_OUT = 159, 145, 133, 33
L_EYE_TOP, L_EYE_BOT, L_EYE_IN, L_EYE_OUT = 386, 374, 362, 263
BLINK_MIN = 0.12       # lid-open / eye-width below this = blink, hold last value
GY_ALPHA = 0.45        # EMA smoothing for the derived gaze_y

# Appearance-based vertical eye-gaze model (ONNX, MobileGaze/L2CS family). We run
# it on a face crop and use its pitch (up/down) output as the "vgaze" signal.
# mobileone_s0 is the fast default (~22 ms/frame CPU, faster with CoreML on a
# Mac); resnet34 is the most accurate but heavier. Weights are pulled on first
# run from the yakhyo/gaze-estimation release.
GAZE_MODEL_DEFAULT = "mobileone_s0"
GAZE_MODELS = ("mobileone_s0", "resnet18", "resnet34")
GAZE_MODEL_URL = (
    "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"
    "{name}_gaze.onnx"
)
FACE_CROP_MARGIN = 0.15  # expand the landmark bbox before cropping for the model
MIN_CROP_PX = 24         # skip the gaze read if the face crop is smaller than this

# One-Euro filter params (smooth when still, snappy when you move). Tuned for
# ~30 fps. Head direction tolerates a higher cutoff; the noisier eye signal
# gets more smoothing.
HEAD_MIN_CUTOFF, HEAD_BETA = 1.2, 0.35
EYE_MIN_CUTOFF, EYE_BETA = 0.8, 0.25

# Calibration averages this many recent frames, so one noisy frame (or a
# mid-saccade sample) can't poison an anchor. ~0.4 s at 30 fps.
CALIB_FRAMES = 12

# Bump when the head/eye signal definition changes so stale calibration (stored
# in the old units) is discarded instead of silently mixed with the new signal.
# v3: "points" is {zone: [[yaw,pitch], ...]} (multiple samples per monitor).
# v4: eye gaze switched from an iris-corner ratio to the ONNX gaze model, so the
#     "sub" (top/bottom) calibration changes units -- but "points" (head pose) is
#     unchanged, so the v3->v4 upgrade keeps monitors and clears only the splits.
CONFIG_VERSION = 4

ZONE_NAMES = {0: "LAPTOP", 1: "TOP", 2: "RIGHT"}
# Short labels for the calibration dots in the preview.
ZONE_TAGS = {0: "L", 1: "T", 2: "R"}
SPAN_FLOOR = 0.05  # avoid divide-by-zero when monitors share an axis


class OneEuro:
    """One-Euro filter: smooths a noisy scalar with low latency.

    Adapts to motion -- heavy smoothing when the signal is still (kills jitter),
    light smoothing when it's moving fast (stays responsive). Far better than a
    fixed-alpha EMA for an interactive pointing signal.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class GazeEstimationONNX:
    """Appearance-based gaze estimator (ONNX). Returns (yaw, pitch) in radians.

    Wraps a MobileGaze/L2CS-style model: the network outputs binned logits that
    we soft-decode (softmax expectation) into continuous angles. We only consume
    pitch (vertical) for the window split, but yaw is decoded too for parity with
    the upstream model. The input size and IO names are read from the model, so
    swapping backbones (mobileone_s0 / resnet18 / resnet34) just works.
    """

    def __init__(self, model_path, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self._bins, self._binwidth, self._angle_offset = 90, 4, 180
        self.idx = np.arange(self._bins, dtype=np.float32)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        hw = inp.shape[2:]  # NCHW -> [H, W]
        try:
            self.input_size = (int(hw[1]), int(hw[0]))  # cv2 wants (w, h)
        except (TypeError, ValueError, IndexError):
            self.input_size = (448, 448)
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0).astype(np.float32)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def estimate(self, face_bgr):
        out = self.session.run(
            self.out_names, {self.input_name: self._preprocess(face_bgr)})
        yaw = np.sum(self._softmax(out[0]) * self.idx, axis=1) \
            * self._binwidth - self._angle_offset
        pitch = np.sum(self._softmax(out[1]) * self.idx, axis=1) \
            * self._binwidth - self._angle_offset
        return float(np.radians(yaw[0])), float(np.radians(pitch[0]))


def gaze_providers():
    """ONNX Runtime providers, best-available first: CoreML (Mac) / CUDA / CPU."""
    order = ["CoreMLExecutionProvider", "CUDAExecutionProvider",
             "CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    sel = [p for p in order if p in avail]
    return sel or ["CPUExecutionProvider"]


def load_config():
    defaults = {"points": {}, "sub": {}, "margin": 0.18, "version": None}
    try:
        with open(CONFIG_PATH) as f:
            defaults.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def save_config(cfg):
    os.makedirs(GAZE_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def reconcile_version(cfg):
    """Migrate calibration to the current CONFIG_VERSION. Returns True if changed.

    v3 -> v4 only changed the eye signal (iris ratio -> ONNX gaze), so we keep
    the head-pose "points" and clear just the "sub" splits. Any other (older or
    unknown) version is cleared wholesale, since its units can't be trusted.
    """
    v = cfg.get("version")
    if v == CONFIG_VERSION:
        return False
    if v == 3:
        had_sub = bool(cfg.get("sub"))
        cfg["sub"] = {}
        cfg["version"] = CONFIG_VERSION
        if had_sub:
            print("Eye gaze upgraded to the ONNX model -> old top/bottom "
                  "calibration cleared (monitor calibration kept). "
                  "Recalibrate splits with t/b.")
        return True
    if cfg.get("points") or cfg.get("sub"):
        print("Calibration format upgraded -> old calibration cleared. "
              "Recalibrate: 1/2/3 for monitors, then t/b per split.")
    cfg["points"], cfg["sub"], cfg["version"] = {}, {}, CONFIG_VERSION
    return True


def write_state(zone, yaw, pitch, vgaze, gaze_y, has_sub,
                auto_focus=False, auto_focus_seconds=3.0):
    os.makedirs(GAZE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "zone": zone,
                "yaw": round(float(yaw), 4),
                "pitch": round(float(pitch), 4),
                "vgaze": round(float(vgaze), 4) if vgaze is not None else None,
                "gaze_y": round(float(gaze_y), 4) if gaze_y is not None else None,
                "has_sub": bool(has_sub),
                # Hands-free dwell focus, driven by the --auto-focus flag. Hammerspoon
                # reads these and commits focus to a steadily-looked-at window.
                "auto_focus": bool(auto_focus),
                "auto_focus_seconds": round(float(auto_focus_seconds), 3),
                "ts": time.time(),
            },
            f,
        )
    os.replace(tmp, STATE_PATH)


def ensure_model():
    """Download the FaceLandmarker model bundle on first run."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    os.makedirs(GAZE_DIR, exist_ok=True)
    print(f"downloading face landmark model -> {MODEL_PATH} ...")
    tmp = MODEL_PATH + ".tmp"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    print("model ready.")
    return MODEL_PATH


def ensure_gaze_model(name):
    """Download the chosen ONNX gaze model on first run."""
    path = os.path.join(GAZE_DIR, f"{name}_gaze.onnx")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(GAZE_DIR, exist_ok=True)
    url = GAZE_MODEL_URL.format(name=name)
    print(f"downloading gaze model '{name}' -> {path} ...")
    tmp = path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print("gaze model ready.")
    return path


def head_dir_from_matrix(matrix):
    """Head forward direction (hx, hy) from the model's 3D facial-transform matrix.

    This is the model's real estimate of head orientation in 3D, so it's far more
    stable than 2D landmark ratios and robust to where you sit. We rotate the
    face's forward axis into camera space and use its x (left/right turn) and y
    (up/down tilt) components as the signal. Exact sign/scale don't matter --
    calibration is nearest-neighbor, so it only needs to be consistent.
    Returns None if the matrix is missing or malformed (caller falls back).
    """
    m = np.asarray(matrix, dtype=float)
    if m.size == 16:
        m = m.reshape(4, 4)
    if m.shape != (4, 4):
        return None
    fwd = m[:3, :3] @ np.array([0.0, 0.0, -1.0])
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return None
    fwd = fwd / n
    return float(fwd[0]), float(fwd[1])


def head_signals(landmarks):
    """Fallback (yaw, pitch) from 2D landmarks, used only if the 3D matrix is
    unavailable in this MediaPipe build.

    yaw:   nose offset between the cheeks, ~[-1, 1]. + = turned toward one side.
    pitch: nose height between the eye line and the chin, ~[0, 1].
           larger = chin tucked / looking up, smaller = looking down.
    """
    nose = landmarks[NOSE]
    lc, rc = landmarks[LEFT_CHEEK], landmarks[RIGHT_CHEEK]
    cx = (lc.x + rc.x) / 2.0
    half_w = (rc.x - lc.x) / 2.0
    yaw = (nose.x - cx) / half_w if abs(half_w) > 1e-6 else 0.0

    eye_y = (landmarks[RIGHT_EYE_OUTER].y + landmarks[LEFT_EYE_OUTER].y) / 2.0
    chin_y = landmarks[CHIN].y
    face_h = chin_y - eye_y
    pitch = (nose.y - eye_y) / face_h if abs(face_h) > 1e-6 else 0.0
    return yaw, pitch


def bbox_from_landmarks(landmarks, w, h, margin=FACE_CROP_MARGIN):
    """Pixel face bounding box from the normalized landmarks, padded by `margin`.

    Gives the ONNX gaze model a head crop without pulling in a separate face
    detector -- the landmarks already bound the face tightly, and the margin adds
    a little context (forehead/jaw) the way the model's training crops did.
    """
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * margin
    x1 += bw * margin
    y0 -= bh * margin
    y1 += bh * margin
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(w, x1))
    y1 = int(min(h, y1))
    return x0, y0, x1, y1


def is_blinking(landmarks):
    """True if the eyes are too closed for a trustworthy gaze read.

    Uses lid-open height over eye width, averaged across both eyes. During a
    blink/squint we hold the last good gaze value instead of feeding the model a
    half-shut eye. Returns False (assume open) if the lid landmarks are missing.
    """
    if len(landmarks) <= max(L_EYE_BOT, R_EYE_BOT, L_EYE_OUT, R_EYE_OUT):
        return False

    def ratio(top_i, bot_i, in_i, out_i):
        width = abs(landmarks[out_i].x - landmarks[in_i].x)
        if width < 1e-6:
            return None
        return (landmarks[bot_i].y - landmarks[top_i].y) / width

    rs = [r for r in (ratio(R_EYE_TOP, R_EYE_BOT, R_EYE_IN, R_EYE_OUT),
                      ratio(L_EYE_TOP, L_EYE_BOT, L_EYE_IN, L_EYE_OUT))
          if r is not None]
    if not rs:
        return False
    return (sum(rs) / len(rs)) < BLINK_MIN


def has_sub(zone, cfg):
    sub = cfg.get("sub", {}).get(str(zone))
    return bool(sub and "top" in sub and "bottom" in sub)


def sub_gaze_y(zone, pitch, vgaze, cfg):
    """Continuous vertical position within a monitor's split, in [0, 1].

    Uses a 2-point calibration (top window, bottom window). Each calibrated
    point is a (pitch, vgaze) pair, so the projection picks up whichever you
    actually move -- your eyes (vgaze), your head (pitch), or both. We project
    the current (pitch, vgaze) onto the top->bottom line and clamp.
    Returns None if the zone has no eye calibration or the eyes are closed.
    """
    if not has_sub(zone, cfg) or vgaze is None:
        return None
    tp = cfg["sub"][str(zone)]["top"]
    bp = cfg["sub"][str(zone)]["bottom"]
    dx, dy = bp[0] - tp[0], bp[1] - tp[1]
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return None
    t = ((pitch - tp[0]) * dx + (vgaze - tp[1]) * dy) / denom
    return float(min(1.0, max(0.0, t)))


def _samples(zone_pts):
    """A zone's stored value as a list of [yaw, pitch] samples.

    Tolerates the old single-point shape [yaw, pitch] (auto-cleared on version
    bump, but cheap to stay robust) by wrapping it in a one-element list.
    """
    if zone_pts and isinstance(zone_pts[0], (list, tuple)):
        return zone_pts
    return [zone_pts]


def _spans(points):
    flat = [s for zp in points.values() for s in _samples(zp)]
    ys = [s[0] for s in flat]
    ps = [s[1] for s in flat]
    yaw_span = max(max(ys) - min(ys), SPAN_FLOOR)
    pitch_span = max(max(ps) - min(ps), SPAN_FLOOR)
    return yaw_span, pitch_span


def distances(yaw, pitch, points):
    """Normalized distance from (yaw, pitch) to each calibrated zone.

    Each axis is scaled by how far apart the calibrated monitors are on that
    axis, so the axis that actually separates your monitors carries the weight.
    A zone can hold several pose samples; its distance is the NEAREST sample, so
    recording a monitor from multiple postures only ever helps.
    """
    yaw_span, pitch_span = _spans(points)
    out = {}
    for k, zp in points.items():
        best = min(
            math.hypot((yaw - py) / yaw_span, (pitch - pp) / pitch_span)
            for (py, pp) in _samples(zp)
        )
        out[int(k)] = best
    return out


def classify(yaw, pitch, cfg, current):
    """Nearest calibrated zone, with a stickiness margin for hysteresis."""
    points = cfg["points"]
    if len(points) < 3:
        return -1  # not calibrated yet
    d = distances(yaw, pitch, points)
    nearest = min(d, key=d.get)
    if current in d and current != nearest:
        # only leave the current zone if another is closer by > margin
        if d[current] - d[nearest] < cfg["margin"]:
            return current
    return nearest


def _to_px(hx, hy, box):
    bx, by, bw, bh = box
    xn = np.clip((hx + 0.7) / 1.4, 0.0, 1.0)         # ~[-0.7, 0.7] -> 0..1
    yn = np.clip((hy + 0.7) / 1.4, 0.0, 1.0)         # look up (neg) -> toward top
    px = int(bx + xn * bw)
    py = int(by + yn * bh)
    return px, py


def draw_overlay(frame, yaw, pitch, cfg, zone, vgaze=None, gaze_y=None):
    h, w = frame.shape[:2]
    points = cfg["points"]
    ready = len(points) >= 3

    label = ZONE_NAMES.get(zone, "—")
    cv2.putText(frame, f"ZONE: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 255), 3)

    # Vertical eye-gaze bar (only meaningful once a split is calibrated).
    sub = cfg.get("sub", {}).get(str(zone), {})
    have_top, have_bot = "top" in sub, "bottom" in sub
    bar_x, bar_y, bar_h = 30, 130, 220
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 16, bar_y + bar_h),
                  (90, 90, 90), 1)
    cv2.putText(frame, "T" if have_top else "t", (bar_x - 2, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 220, 255) if have_top else (120, 120, 120), 2)
    cv2.putText(frame, "B" if have_bot else "b", (bar_x - 2, bar_y + bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 220, 255) if have_bot else (120, 120, 120), 2)
    if gaze_y is not None:
        gy_py = int(bar_y + float(gaze_y) * bar_h)
        cv2.rectangle(frame, (bar_x - 4, gy_py - 3), (bar_x + 20, gy_py + 3),
                      (80, 255, 120), -1)
        cv2.putText(frame, f"gaze_y {gaze_y:.2f}", (bar_x + 26, gy_py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 1)
    elif vgaze is not None and (have_top or have_bot):
        cv2.putText(frame, "look at the other window + press t/b",
                    (bar_x + 26, bar_y + bar_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 2D pose box (yaw on x, pitch on y)
    box = (w - 240, 70, 200, 200)
    bx, by, bw, bh = box
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
    cv2.line(frame, (bx + bw // 2, by), (bx + bw // 2, by + bh), (60, 60, 60), 1)
    cv2.line(frame, (bx, by + bh // 2), (bx + bw, by + bh // 2), (60, 60, 60), 1)

    # calibrated points (one ring per recorded sample; label the centroid)
    total_samples = 0
    for k, zp in points.items():
        samples = _samples(zp)
        total_samples += len(samples)
        hit = (int(k) == zone)
        col = (60, 220, 255) if hit else (160, 160, 160)
        for (py, pp) in samples:
            cpx, cpy = _to_px(py, pp, box)
            cv2.circle(frame, (cpx, cpy), 5, col, 1)
        mx = sum(s[0] for s in samples) / len(samples)
        my = sum(s[1] for s in samples) / len(samples)
        lpx0, lpy0 = _to_px(mx, my, box)
        cv2.putText(frame, ZONE_TAGS[int(k)], (lpx0 - 6, lpy0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # live pose dot
    lpx, lpy = _to_px(yaw, pitch, box)
    cv2.circle(frame, (lpx, lpy), 6, (80, 255, 120), -1)

    if not ready:
        msg = "CALIBRATE: look at each monitor and press 1=laptop  2=top  3=right"
        cv2.putText(frame, msg, (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 255), 2)
    else:
        cv2.putText(frame,
                    "1/2/3=add sample  u=undo  t/b=top/bottom window  x=clear eye  r=reset  [ ]=sticky  q=quit",
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame,
                f"zones: {len(points)}/3   samples: {total_samples}   margin={cfg['margin']:.2f}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)


def main():
    ap = argparse.ArgumentParser(description="Head-pose monitor selector")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--no-preview", action="store_true", help="run headless, no window")
    ap.add_argument("--fps", type=float, default=30.0, help="max processing fps")
    ap.add_argument("--gaze-model", choices=GAZE_MODELS, default=GAZE_MODEL_DEFAULT,
                    help="ONNX eye-gaze model: mobileone_s0 (fast, default), "
                         "resnet18, or resnet34 (most accurate, heavier on CPU)")
    ap.add_argument("--gaze-stride", type=int, default=1,
                    help="run the gaze model every Nth frame (raise to save CPU; "
                         "head pose still runs every frame)")
    ap.add_argument("--auto-focus", action="store_true",
                    help="hands-free dwell focus: Hammerspoon commits focus to a "
                         "window once you've looked at it steadily for the dwell "
                         "time -- no trigger press needed (off by default)")
    ap.add_argument("--auto-focus-seconds", type=float, default=3.0,
                    help="dwell time in seconds before --auto-focus commits "
                         "(default 3)")
    args = ap.parse_args()

    cfg = load_config()
    if reconcile_version(cfg):
        save_config(cfg)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    model_path = ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    ts_ms = 0

    gaze_path = ensure_gaze_model(args.gaze_model)
    providers = gaze_providers()
    engine = GazeEstimationONNX(gaze_path, providers)
    gaze_stride = max(1, args.gaze_stride)
    print(f"gaze model: {args.gaze_model}  input={engine.input_size}  "
          f"providers={providers}  stride={gaze_stride}")

    zone = -1
    last_yaw, last_pitch = 0.0, 0.0
    vgaze_s = None   # smoothed eye-gaze pitch (None until first good read)
    gaze_y_s = None  # smoothed within-monitor vertical position
    f_hx = OneEuro(HEAD_MIN_CUTOFF, HEAD_BETA)
    f_hy = OneEuro(HEAD_MIN_CUTOFF, HEAD_BETA)
    f_vg = OneEuro(EYE_MIN_CUTOFF, EYE_BETA)
    recent = deque(maxlen=CALIB_FRAMES)  # recent (yaw, pitch, vgaze) for calibration
    last_added = None  # zone of the most recently added pose sample (for undo)
    frame_i = 0
    min_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    print(f"gaze-focus running. writing -> {STATE_PATH}")
    if args.auto_focus:
        print(f"auto-focus ON: focus commits after a steady "
              f"{args.auto_focus_seconds:.1f}s look (no trigger needed).")
    if len(cfg["points"]) < 3:
        print("Not calibrated yet. Look at each monitor and press 1=laptop, 2=top, 3=right.")
    else:
        print("Calibrated. Turn/tilt toward each monitor and watch the ZONE label.")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror so preview feels natural
            h, w = frame.shape[:2]
            frame_i += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms += 1
            result = landmarker.detect_for_video(mp_image, ts_ms)

            yaw, pitch = last_yaw, last_pitch
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                hd = None
                if result.facial_transformation_matrixes:
                    hd = head_dir_from_matrix(
                        result.facial_transformation_matrixes[0])
                if hd is None:
                    hd = head_signals(lm)  # 2D fallback if no 3D matrix
                yaw = f_hx(hd[0], t0)
                pitch = f_hy(hd[1], t0)
                last_yaw, last_pitch = yaw, pitch

                # Vertical eye gaze from the ONNX model. Skip during a blink or on
                # a strided frame -> vgaze_s holds its last good value, so the
                # split pick doesn't jump when the eyes are shut or we're saving
                # CPU. Mirroring the frame flips yaw (which we ignore), not pitch.
                if not is_blinking(lm) and frame_i % gaze_stride == 0:
                    x0, y0, x1, y1 = bbox_from_landmarks(lm, w, h)
                    if (x1 - x0) >= MIN_CROP_PX and (y1 - y0) >= MIN_CROP_PX:
                        _, gaze_pitch = engine.estimate(frame[y0:y1, x0:x1])
                        vgaze_s = f_vg(gaze_pitch, t0)
                recent.append((yaw, pitch, vgaze_s))

            zone = classify(yaw, pitch, cfg, zone)
            gy = sub_gaze_y(zone, pitch, vgaze_s, cfg)
            if gy is not None:
                gaze_y_s = gy if gaze_y_s is None else (
                    GY_ALPHA * gy + (1.0 - GY_ALPHA) * gaze_y_s)
            else:
                gaze_y_s = None
            write_state(zone, yaw, pitch, vgaze_s, gaze_y_s, has_sub(zone, cfg),
                        args.auto_focus, args.auto_focus_seconds)

            if not args.no_preview:
                draw_overlay(frame, yaw, pitch, cfg, zone, vgaze_s, gaze_y_s)
                cv2.imshow("gaze-focus (head pose)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("1"), ord("2"), ord("3")):
                    z = key - ord("1")  # '1'->0, '2'->1, '3'->2
                    head_pts = [(a, b) for (a, b, _) in recent]
                    if head_pts:
                        ax = sum(p[0] for p in head_pts) / len(head_pts)
                        ay = sum(p[1] for p in head_pts) / len(head_pts)
                        samples = cfg["points"].setdefault(str(z), [])
                        samples.append([round(ax, 4), round(ay, 4)])
                        last_added = z
                        save_config(cfg)
                        print(f"added {ZONE_NAMES[z]} sample #{len(samples)} "
                              f"(avg of {len(head_pts)} frames): "
                              f"yaw={ax:.3f} pitch={ay:.3f}")
                    else:
                        print("no face detected yet -- look at the camera first.")
                elif key == ord("u"):
                    if last_added is not None and cfg["points"].get(str(last_added)):
                        dropped = cfg["points"][str(last_added)].pop()
                        if not cfg["points"][str(last_added)]:
                            cfg["points"].pop(str(last_added))
                        save_config(cfg)
                        print(f"undid last {ZONE_NAMES[last_added]} sample "
                              f"(yaw={dropped[0]:.3f} pitch={dropped[1]:.3f}).")
                        last_added = None
                    else:
                        print("nothing to undo.")
                elif key in (ord("t"), ord("b")):
                    eye_pts = [(b, c) for (_, b, c) in recent if c is not None]
                    if zone in (0, 1, 2) and eye_pts:
                        slot = "top" if key == ord("t") else "bottom"
                        ap_ = sum(p[0] for p in eye_pts) / len(eye_pts)
                        av = sum(p[1] for p in eye_pts) / len(eye_pts)
                        cfg.setdefault("sub", {}).setdefault(str(zone), {})[slot] = [
                            round(ap_, 4), round(av, 4)]
                        save_config(cfg)
                        print(f"recorded {slot.upper()} window of {ZONE_NAMES[zone]} "
                              f"(avg of {len(eye_pts)}): pitch={ap_:.3f} vgaze={av:.3f}")
                    elif not eye_pts:
                        print("can't record: eyes not detected (blink/lighting).")
                    else:
                        print("look at a calibrated monitor first (zone unknown).")
                elif key == ord("x"):
                    if str(zone) in cfg.get("sub", {}):
                        cfg["sub"].pop(str(zone), None)
                        save_config(cfg)
                        print(f"cleared eye calibration for {ZONE_NAMES.get(zone, zone)}.")
                elif key == ord("r"):
                    cfg["points"] = {}
                    cfg["sub"] = {}
                    save_config(cfg)
                    print("calibration cleared (monitors + eye).")
                elif key == ord("["):
                    cfg["margin"] = round(min(0.6, cfg["margin"] + 0.02), 3)
                    save_config(cfg)
                    print(f"margin={cfg['margin']:.2f} (stickier)")
                elif key == ord("]"):
                    cfg["margin"] = round(max(0.0, cfg["margin"] - 0.02), 3)
                    save_config(cfg)
                    print(f"margin={cfg['margin']:.2f} (looser)")

            dt = time.time() - t0
            if dt < min_dt:
                time.sleep(min_dt - dt)
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
