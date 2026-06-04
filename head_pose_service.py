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
your eyes for that, not your head. So we also read vertical EYE gaze from the
iris landmarks (measured against the rigid eye corners, so it tracks eyeball
rotation rather than eyelid movement). Calibrate a screen's split by looking at
the top window (press 't') and the bottom window (press 'b') -- hold your gaze
for a beat so it averages cleanly; from then on we emit a continuous
"gaze_y" in [0,1] (0 = top, 1 = bottom) whenever that screen is the active zone.
Hammerspoon uses gaze_y to focus the exact window at that height. This works for
any vertical split (2, 3, ... windows), not just two.

It NEVER changes window focus. It just keeps the current zone (and, when
calibrated, gaze_y) live. The commit (actually switching focus) is done by
Hammerspoon when you press your trigger (mouse4 or a hotkey). Eyes point, the
trigger commits.

Live keys in the preview window:
    1  -> record current pose as the LAPTOP monitor
    2  -> record current pose as the TOP monitor
    3  -> record current pose as the RIGHT monitor
    t  -> record TOP window of the current monitor (look at it first)
    b  -> record BOTTOM window of the current monitor (look at it first)
    x  -> clear the current monitor's top/bottom (eye) calibration
    r  -> reset / clear ALL calibration (monitors + eye)
    [  -> stickier (harder to switch zones)
    ]  -> looser (easier to switch zones)
    q / ESC -> quit
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

# Iris + eyelid landmarks for vertical eye gaze. The 478-landmark model includes
# iris points (468-477) by default, so no extra options are needed.
# Per eye we use upper/lower lid, inner/outer corner, and the iris center.
R_EYE_TOP, R_EYE_BOT, R_EYE_IN, R_EYE_OUT, R_IRIS = 159, 145, 133, 33, 473
L_EYE_TOP, L_EYE_BOT, L_EYE_IN, L_EYE_OUT, L_IRIS = 386, 374, 362, 263, 468
BLINK_MIN = 0.12       # lid-open / eye-width below this = blink, hold last value
GY_ALPHA = 0.45        # EMA smoothing for the derived gaze_y

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
CONFIG_VERSION = 2

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


def write_state(zone, yaw, pitch, vgaze, gaze_y, has_sub):
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


def eye_vgaze(landmarks):
    """Vertical eye-gaze signal: more positive = looking further down.

    For each eye, measures the iris height relative to the eye-corner line
    (which is rigid and moves with the head, not the lid), scaled by eye width,
    then averages both eyes. This is more faithful to actual eyeball rotation --
    and far less coupled to blink/lid movement -- than a lid-relative measure.
    Returns None during a blink/squint (lids too close) so the caller can hold
    the last good value instead of jumping. Also None if the model build lacks
    iris landmarks, which disables eye gaze without breaking head pose.
    """
    if len(landmarks) <= max(L_IRIS, R_IRIS):
        return None

    def one(top_i, bot_i, in_i, out_i, iris_i):
        open_h = landmarks[bot_i].y - landmarks[top_i].y
        width = abs(landmarks[out_i].x - landmarks[in_i].x)
        if width < 1e-6 or (open_h / width) < BLINK_MIN:
            return None  # eye closed / squinting -> unreliable
        corner_mid_y = (landmarks[in_i].y + landmarks[out_i].y) / 2.0
        return (landmarks[iris_i].y - corner_mid_y) / width

    vals = [
        v for v in (
            one(R_EYE_TOP, R_EYE_BOT, R_EYE_IN, R_EYE_OUT, R_IRIS),
            one(L_EYE_TOP, L_EYE_BOT, L_EYE_IN, L_EYE_OUT, L_IRIS),
        ) if v is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


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


def _spans(points):
    ys = [p[0] for p in points.values()]
    ps = [p[1] for p in points.values()]
    yaw_span = max(max(ys) - min(ys), SPAN_FLOOR)
    pitch_span = max(max(ps) - min(ps), SPAN_FLOOR)
    return yaw_span, pitch_span


def distances(yaw, pitch, points):
    """Normalized distance from (yaw, pitch) to each calibrated zone point.

    Each axis is scaled by how far apart the calibrated monitors are on that
    axis, so the axis that actually separates your monitors carries the weight.
    """
    yaw_span, pitch_span = _spans(points)
    out = {}
    for k, (py, pp) in points.items():
        dy = (yaw - py) / yaw_span
        dp = (pitch - pp) / pitch_span
        out[int(k)] = math.hypot(dy, dp)
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

    # calibrated points
    for k, (py, pp) in points.items():
        cpx, cpy = _to_px(py, pp, box)
        hit = (int(k) == zone)
        col = (60, 220, 255) if hit else (160, 160, 160)
        cv2.circle(frame, (cpx, cpy), 7, col, 2)
        cv2.putText(frame, ZONE_TAGS[int(k)], (cpx - 6, cpy - 10),
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
                    "1/2/3=monitors  t/b=top/bottom window  x=clear eye  r=reset  [ ]=sticky  q=quit",
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"calibrated: {len(points)}/3   margin={cfg['margin']:.2f}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)


def main():
    ap = argparse.ArgumentParser(description="Head-pose monitor selector")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--no-preview", action="store_true", help="run headless, no window")
    ap.add_argument("--fps", type=float, default=30.0, help="max processing fps")
    args = ap.parse_args()

    cfg = load_config()
    if cfg.get("version") != CONFIG_VERSION:
        if cfg.get("points") or cfg.get("sub"):
            print("Signal model upgraded -> old calibration cleared. "
                  "Recalibrate: press 1/2/3 for monitors, then t/b per split.")
        cfg["points"], cfg["sub"], cfg["version"] = {}, {}, CONFIG_VERSION
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

    zone = -1
    last_yaw, last_pitch = 0.0, 0.0
    vgaze_s = None   # smoothed raw eye-gaze
    gaze_y_s = None  # smoothed within-monitor vertical position
    f_hx = OneEuro(HEAD_MIN_CUTOFF, HEAD_BETA)
    f_hy = OneEuro(HEAD_MIN_CUTOFF, HEAD_BETA)
    f_vg = OneEuro(EYE_MIN_CUTOFF, EYE_BETA)
    recent = deque(maxlen=CALIB_FRAMES)  # recent (yaw, pitch, vgaze) for calibration
    min_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    print(f"gaze-focus running. writing -> {STATE_PATH}")
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
                raw_v = eye_vgaze(lm)
                if raw_v is not None:
                    vgaze_s = f_vg(raw_v, t0)
                recent.append((yaw, pitch, vgaze_s))

            zone = classify(yaw, pitch, cfg, zone)
            gy = sub_gaze_y(zone, pitch, vgaze_s, cfg)
            if gy is not None:
                gaze_y_s = gy if gaze_y_s is None else (
                    GY_ALPHA * gy + (1.0 - GY_ALPHA) * gaze_y_s)
            else:
                gaze_y_s = None
            write_state(zone, yaw, pitch, vgaze_s, gaze_y_s, has_sub(zone, cfg))

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
                        cfg["points"][str(z)] = [round(ax, 4), round(ay, 4)]
                        save_config(cfg)
                        print(f"recorded {ZONE_NAMES[z]} (avg of {len(head_pts)} "
                              f"frames): yaw={ax:.3f} pitch={ay:.3f}")
                    else:
                        print("no face detected yet -- look at the camera first.")
                elif key in (ord("t"), ord("b")):
                    eye_pts = [(b, c) for (_, b, c) in recent if c is not None]
                    if zone in (0, 1, 2) and eye_pts:
                        slot = "top" if key == ord("t") else "bottom"
                        ap = sum(p[0] for p in eye_pts) / len(eye_pts)
                        av = sum(p[1] for p in eye_pts) / len(eye_pts)
                        cfg.setdefault("sub", {}).setdefault(str(zone), {})[slot] = [
                            round(ap, 4), round(av, 4)]
                        save_config(cfg)
                        print(f"recorded {slot.upper()} window of {ZONE_NAMES[zone]} "
                              f"(avg of {len(eye_pts)}): pitch={ap:.3f} vgaze={av:.3f}")
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
