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
number key (1/2/3). That records the head pose (yaw, pitch) for that monitor.
From then on, each frame is classified to whichever calibrated point it's
closest to. No hand-tuned thresholds, no assumptions about geometry -- it just
learns where you point your head for each screen. A stickiness margin keeps the
zone stable so it doesn't flicker at the boundaries.

EYE GAZE (sub-window selection):
Head pose can pick the monitor, but it can't tell apart two windows split on
the SAME screen (e.g. a top + bottom split on the vertical monitor) -- you move
your eyes for that, not your head. So we also read vertical EYE gaze from the
iris landmarks. Calibrate a screen's split by looking at the top window (press
't') and the bottom window (press 'b'); from then on we emit a continuous
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
VGAZE_ALPHA = 0.35     # EMA smoothing for the (noisy) raw eye-gaze signal
GY_ALPHA = 0.45        # EMA smoothing for the derived gaze_y

ZONE_NAMES = {0: "LAPTOP", 1: "TOP", 2: "RIGHT"}
# Short labels for the calibration dots in the preview.
ZONE_TAGS = {0: "L", 1: "T", 2: "R"}
SPAN_FLOOR = 0.05  # avoid divide-by-zero when monitors share an axis


def load_config():
    defaults = {"points": {}, "sub": {}, "margin": 0.18}
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


def head_signals(landmarks):
    """Return (yaw, pitch).

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
    """Vertical eye-gaze ratio, ~[0, 1]: 0 = looking up, 1 = looking down.

    For each eye, measures where the iris center sits between the upper and
    lower lids, normalized by how open the eye is, then averages both eyes.
    Returns None during a blink/squint (lids too close) so the caller can hold
    the last good value instead of jumping. Also None if the model build lacks
    iris landmarks, which disables eye gaze without breaking head pose.
    """
    if len(landmarks) <= max(L_IRIS, R_IRIS):
        return None

    def one(top_i, bot_i, in_i, out_i, iris_i):
        top = landmarks[top_i].y
        bot = landmarks[bot_i].y
        open_h = bot - top
        width = abs(landmarks[out_i].x - landmarks[in_i].x)
        if width < 1e-6 or (open_h / width) < BLINK_MIN:
            return None
        return (landmarks[iris_i].y - top) / open_h

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


def _to_px(yaw, pitch, box):
    bx, by, bw, bh = box
    yn = np.clip((yaw + 1.0) / 2.0, 0.0, 1.0)        # -1..1 -> 0..1
    pn = np.clip((pitch - 0.2) / 0.6, 0.0, 1.0)      # 0.2..0.8 -> 0..1
    px = int(bx + yn * bw)
    py = int(by + (1.0 - pn) * bh)                   # look up -> toward top
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    model_path = ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    ts_ms = 0

    zone = -1
    last_yaw, last_pitch = 0.0, 0.5
    vgaze_s = None   # smoothed raw eye-gaze
    gaze_y_s = None  # smoothed within-monitor vertical position
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
                yaw, pitch = head_signals(lm)
                last_yaw, last_pitch = yaw, pitch
                raw_v = eye_vgaze(lm)
                if raw_v is not None:
                    vgaze_s = raw_v if vgaze_s is None else (
                        VGAZE_ALPHA * raw_v + (1.0 - VGAZE_ALPHA) * vgaze_s)

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
                    cfg["points"][str(z)] = [round(last_yaw, 4), round(last_pitch, 4)]
                    save_config(cfg)
                    print(f"recorded {ZONE_NAMES[z]}: yaw={last_yaw:.3f} pitch={last_pitch:.3f}")
                elif key in (ord("t"), ord("b")):
                    if zone in (0, 1, 2) and vgaze_s is not None:
                        slot = "top" if key == ord("t") else "bottom"
                        cfg.setdefault("sub", {}).setdefault(str(zone), {})[slot] = [
                            round(last_pitch, 4), round(vgaze_s, 4)]
                        save_config(cfg)
                        print(f"recorded {slot.upper()} window of {ZONE_NAMES[zone]}: "
                              f"pitch={last_pitch:.3f} vgaze={vgaze_s:.3f}")
                    elif vgaze_s is None:
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
