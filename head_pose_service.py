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

It NEVER changes window focus. It just keeps the current zone live. The commit
(actually switching focus) is done by Hammerspoon when you press your trigger
(mouse4 or a hotkey). Head points, the trigger commits.

Live keys in the preview window:
    1  -> record current pose as the LAPTOP monitor
    2  -> record current pose as the TOP monitor
    3  -> record current pose as the RIGHT monitor
    r  -> reset / clear all calibration
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

ZONE_NAMES = {0: "LAPTOP", 1: "TOP", 2: "RIGHT"}
# Short labels for the calibration dots in the preview.
ZONE_TAGS = {0: "L", 1: "T", 2: "R"}
SPAN_FLOOR = 0.05  # avoid divide-by-zero when monitors share an axis


def load_config():
    defaults = {"points": {}, "margin": 0.18}
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


def write_state(zone, yaw, pitch):
    os.makedirs(GAZE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "zone": zone,
                "yaw": round(float(yaw), 4),
                "pitch": round(float(pitch), 4),
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


def draw_overlay(frame, yaw, pitch, cfg, zone):
    h, w = frame.shape[:2]
    points = cfg["points"]
    ready = len(points) >= 3

    label = ZONE_NAMES.get(zone, "—")
    cv2.putText(frame, f"ZONE: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 255), 3)

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
        cv2.putText(frame, "1/2/3=recalibrate  r=reset  [ ]=stickiness  q=quit",
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
                yaw, pitch = head_signals(result.face_landmarks[0])
                last_yaw, last_pitch = yaw, pitch

            zone = classify(yaw, pitch, cfg, zone)
            write_state(zone, yaw, pitch)

            if not args.no_preview:
                draw_overlay(frame, yaw, pitch, cfg, zone)
                cv2.imshow("gaze-focus (head pose)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("1"), ord("2"), ord("3")):
                    z = key - ord("1")  # '1'->0, '2'->1, '3'->2
                    cfg["points"][str(z)] = [round(last_yaw, 4), round(last_pitch, 4)]
                    save_config(cfg)
                    print(f"recorded {ZONE_NAMES[z]}: yaw={last_yaw:.3f} pitch={last_pitch:.3f}")
                elif key == ord("r"):
                    cfg["points"] = {}
                    save_config(cfg)
                    print("calibration cleared.")
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
