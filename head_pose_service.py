#!/usr/bin/env python3
"""
Head-pose monitor selector (gaze-focus).

Reads your webcam, figures out which way your head is turned, and continuously
writes the current "zone" to ~/.gaze/state.json:

    0 = LEFT monitor   1 = CENTER monitor   2 = RIGHT monitor

It NEVER changes window focus. It just keeps the current zone live. The commit
(actually switching focus) is done by Hammerspoon when you press your trigger
(mouse4 or a hotkey). Eyes/head point, the trigger commits.

Signal: horizontal position of the nose between the two cheek landmarks,
normalized to roughly -1 (turned one way) .. +1 (turned the other way). This is
far more robust to lighting than full solvePnP head pose and doesn't flip.

Live keys in the preview window:
    c  -> calibrate: set your current head position as CENTER (neutral)
    i  -> invert left/right (if the zones are mirrored)
    [  -> make it less sensitive (wider center)
    ]  -> make it more sensitive (narrower center)
    q / ESC -> quit
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import mediapipe as mp

HOME = os.path.expanduser("~")
GAZE_DIR = os.path.join(HOME, ".gaze")
STATE_PATH = os.path.join(GAZE_DIR, "state.json")
CONFIG_PATH = os.path.join(GAZE_DIR, "config.json")

# MediaPipe FaceMesh landmark indices
NOSE = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

ZONE_NAMES = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}


def load_config():
    defaults = {"neutral": 0.0, "enter": 0.18, "exit": 0.10, "invert": False}
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


def write_state(zone, offset):
    os.makedirs(GAZE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"zone": zone, "offset": round(float(offset), 4), "ts": time.time()}, f)
    os.replace(tmp, STATE_PATH)


def head_offset(landmarks):
    """Return nose offset in ~[-1, 1]. 0 = centered between cheeks."""
    nose = landmarks[NOSE]
    lc = landmarks[LEFT_CHEEK]
    rc = landmarks[RIGHT_CHEEK]
    center_x = (lc.x + rc.x) / 2.0
    half_width = (rc.x - lc.x) / 2.0
    if half_width <= 1e-6:
        return 0.0
    return (nose.x - center_x) / half_width


def classify(offset, cfg, current_zone):
    """3-zone state machine with hysteresis so boundaries don't flicker."""
    o = offset - cfg["neutral"]
    if cfg["invert"]:
        o = -o
    enter = cfg["enter"]
    exit_ = cfg["exit"]

    if current_zone == 1:  # currently CENTER
        if o <= -enter:
            return 0
        if o >= enter:
            return 2
        return 1
    # currently on a side -> need to come back inside exit band to recenter
    if abs(o) <= exit_:
        return 1
    if o <= -enter:
        return 0
    if o >= enter:
        return 2
    return current_zone


def draw_overlay(frame, offset, cfg, zone):
    h, w = frame.shape[:2]
    o = offset - cfg["neutral"]
    if cfg["invert"]:
        o = -o

    # zone label
    cv2.putText(frame, f"ZONE: {ZONE_NAMES[zone]}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 255), 3)

    # offset bar
    bar_x, bar_y, bar_w, bar_h = 20, h - 60, w - 40, 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    mid = bar_x + bar_w // 2
    cv2.line(frame, (mid, bar_y), (mid, bar_y + bar_h), (120, 120, 120), 1)
    # threshold marks
    for sign in (-1, 1):
        tx = int(mid + sign * cfg["enter"] * (bar_w / 2))
        cv2.line(frame, (tx, bar_y), (tx, bar_y + bar_h), (0, 140, 255), 1)
    # current pos
    pos = int(np.clip(mid + o * (bar_w / 2), bar_x, bar_x + bar_w))
    cv2.circle(frame, (pos, bar_y + bar_h // 2), 9, (60, 220, 255), -1)

    cv2.putText(frame, "c=center  i=invert  [ ]=sensitivity  q=quit",
                (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"enter={cfg['enter']:.2f} invert={cfg['invert']}",
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

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    zone = 1
    last_raw_offset = 0.0
    min_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    print(f"gaze-focus running. writing -> {STATE_PATH}")
    print("Turn to each monitor and check the ZONE label. Press 'c' looking straight ahead to center.")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror so preview feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            offset = last_raw_offset
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                offset = head_offset(lm)
                last_raw_offset = offset

            zone = classify(offset, cfg, zone)
            write_state(zone, offset)

            if not args.no_preview:
                draw_overlay(frame, offset, cfg, zone)
                cv2.imshow("gaze-focus (head pose)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    cfg["neutral"] = last_raw_offset
                    save_config(cfg)
                    print(f"centered: neutral={cfg['neutral']:.3f}")
                elif key == ord("i"):
                    cfg["invert"] = not cfg["invert"]
                    save_config(cfg)
                    print(f"invert={cfg['invert']}")
                elif key == ord("["):
                    cfg["enter"] = min(0.6, cfg["enter"] + 0.02)
                    cfg["exit"] = max(0.03, cfg["enter"] - 0.08)
                    save_config(cfg)
                    print(f"enter={cfg['enter']:.2f}")
                elif key == ord("]"):
                    cfg["enter"] = max(0.06, cfg["enter"] - 0.02)
                    cfg["exit"] = max(0.03, cfg["enter"] - 0.08)
                    save_config(cfg)
                    print(f"enter={cfg['enter']:.2f}")

            dt = time.time() - t0
            if dt < min_dt:
                time.sleep(min_dt - dt)
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
