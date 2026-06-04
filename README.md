# gaze-focus

Turn your head toward a monitor, press a button, and focus jumps to the
frontmost window on that screen. Two pieces:

- **`head_pose_service.py`** — webcam service. Watches which way your head is
  turned and writes the current zone (`0`=left, `1`=center, `2`=right) to
  `~/.gaze/state.json`. It *never* moves focus on its own.
- **`gaze.lua`** — Hammerspoon module. On your trigger (mouse4 or a hotkey) it
  reads the zone and focuses the window on that monitor, with a brief edge flash.

Eyes/head point. The trigger commits. No thrash.

---

## 1. Run the head-pose service

```bash
cd Code/gaze-focus
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python head_pose_service.py
```

First run macOS will ask for **Camera** access for your terminal — allow it.
On first launch it also downloads a small (~3.8 MB) face-landmark model to
`~/.gaze/face_landmarker.task`, so you'll need internet that one time.

A preview window opens. Tune it once:

1. Look straight at your center monitor and press **`c`** to set neutral.
2. Turn your head toward your left and right monitors — watch the **ZONE** label.
   - If left/right are swapped, press **`i`** to invert.
   - Too twitchy or not sensitive enough? `[` widens the center deadzone, `]`
     narrows it.
3. Settings persist to `~/.gaze/config.json`, so you only do this once.

Other keys: `q` / `ESC` quit. Flags: `--camera N`, `--no-preview`, `--fps N`.

## 2. Install Hammerspoon + the commit trigger

1. Install Hammerspoon: `brew install --cask hammerspoon` (or from
   https://www.hammerspoon.org), launch it once.
2. Copy the lua module into place:
   ```bash
   mkdir -p ~/.hammerspoon
   cp gaze.lua ~/.hammerspoon/gaze.lua
   echo 'require("gaze")' >> ~/.hammerspoon/init.lua
   ```
3. Hammerspoon menubar icon → **Reload Config**. You'll see a "gaze-focus
   loaded" toast.
4. Grant permissions when prompted:
   - **Accessibility** (System Settings → Privacy & Security → Accessibility) so
     it can move focus.
   - **Input Monitoring** so it can read the mouse4 button.

## 3. Use it

- **mouse4** (thumb "back" button) → focus the monitor you're facing.
- **⌘⌥⌃G** → same thing via keyboard. Change the binding at the bottom of
  `gaze.lua` to any key/combo you want.

Turn your head, tap the trigger, focus follows. The screen edge flashes cyan so
you get instant "switched here" feedback.

---

## Notes & tuning

- **Why head pose, not eye gaze:** a single webcam can't reliably tell *where*
  on three screens your pupils point, but it reads left/center/right head turn
  cleanly. On a 3-monitor desk you naturally turn your head for the outer ones.
- **Zone ↔ monitor mapping** is purely left-to-right by screen position, so it
  works no matter how macOS numbers your displays.
- **Hysteresis** (separate enter/exit thresholds) keeps the zone stable near
  boundaries, so the zone is settled by the time you press the trigger.
- To swap the trigger to a real **foot pedal** later, just map the pedal to
  ⌘⌥⌃G (or any key) and you're done — no code change.
- Want focus to follow automatically without a trigger? That's the thrashy mode
  we deliberately avoided, but it's a few lines if you change your mind.
