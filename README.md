# gaze-focus

Point your head toward a monitor, press a button, and focus jumps to the window
on that screen — and if a screen has two windows split top/bottom, your **eyes**
pick which one. Two pieces:

- **`head_pose_service.py`** — webcam service. Watches where your head is pointed
  in two axes (left/right turn + up/down tilt) and writes the current zone
  (`0`=laptop, `1`=top, `2`=right) to `~/.gaze/state.json`. It also reads
  vertical **eye gaze** from your irises, so once you calibrate a screen's split
  it emits a continuous `gaze_y` (`0`=top … `1`=bottom). It *never* moves focus
  on its own.
- **`gaze.lua`** — Hammerspoon module. On your trigger (mouse4 or a hotkey) it
  reads the zone, focuses the monitor you're pointed at, and — when `gaze_y` is
  live and that screen holds 2+ windows — the exact window at that height. Brief
  cyan flash around whatever it focused.

Head picks the screen. Eyes pick the window. The trigger commits. No thrash.

**Layout it's tuned for:** a laptop on the desk (you look *down*), a monitor
mounted *above* it (you look *up*), and a monitor off to the *right* (you *turn*).
Laptop and top are both straight ahead, so head turn alone can't separate them —
that's why it also reads up/down tilt. It learns your actual head pose for each
screen via a one-time calibration, so any 3-monitor arrangement works; the labels
are just names.

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

A preview window opens. **Calibrate once** — look at each monitor the way you
naturally would and press its number:

1. Look at your **laptop** → press **`1`**
2. Look at the **top** monitor → press **`2`**
3. Look at the **right** monitor → press **`3`**

After all three are set, the **ZONE** label tracks whichever monitor your head
is closest to. The green dot in the box is your live head pose; the labeled
circles (L/T/R) are your calibrated points. Re-press a number anytime to
re-record it; `r` clears everything.

- Switching too eagerly or not eagerly enough? `[` makes it stickier (harder to
  leave the current zone), `]` makes it looser.
- Calibration persists to `~/.gaze/config.json`, so you only do this once.

### Optional: calibrate a screen's top/bottom split (eye gaze)

For any monitor that holds two windows stacked vertically (e.g. the right
monitor), you can let your eyes pick which one. While the **ZONE** label shows
that monitor:

1. Look at the **top** window → press **`t`**
2. Look at the **bottom** window → press **`b`**

The vertical bar on the left then shows a live `gaze_y` dot moving as you glance
up/down. `x` clears that screen's eye calibration. Keep your head fairly steady
and move your **eyes** between the windows — that's the signal this reads. Eye
gaze is noisier than head pose, so if the pick feels twitchy, re-record `t`/`b`
while looking squarely at each window's center.

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

- **mouse4** (thumb "back" button) → focus the monitor you're pointed at.
- **⌘⌥⌃G** → same thing via keyboard. Change the binding at the bottom of
  `gaze.lua` to any key/combo you want.

Point your head, tap the trigger, focus follows. Whatever it focuses (a screen,
or a single window when eye gaze is live) flashes cyan so you get instant
"switched here" feedback. On a screen with a calibrated top/bottom split it picks
the window your eyes are on; otherwise it focuses the frontmost window there.

`gaze.lua` figures out which physical screen each zone is automatically: the
**right** zone → your rightmost display; of the remaining two, the higher one →
**top**, the lower one → **laptop**. If macOS has your displays arranged so that
guess is wrong, set the screen names in the `OVERRIDE` table at the top of
`gaze.lua` (run `hs.inspect(hs.screen.allScreens())` in the Hammerspoon console
to see the names).

---

## Notes & tuning

- **Head pose for screens, eye gaze for windows:** a single webcam can't reliably
  tell *which of three screens* your pupils point at, but it reads head turn +
  tilt cleanly, and on a spread-out desk you naturally move your head toward each
  screen. *Within* one screen, though, you move your eyes, not your head — so the
  top/bottom-window pick uses iris position instead. Right tool per scale.
- **Two axes (screen pick):** left/right turn (yaw) separates the right monitor;
  up/down tilt (pitch) separates the laptop from the monitor above it. Each frame
  is matched to the *nearest* calibrated head pose — no hand-tuned thresholds, so
  it adapts to wherever you actually point.
- **`gaze_y` (window pick):** derived by projecting your current (head-pitch,
  eye-vertical) onto the line between your calibrated top and bottom samples, so
  it works whether you move your eyes, tip your head, or both. Smoothed and
  blink-guarded; Hammerspoon maps it to the window at that height on the screen.
- **Hysteresis** (the stickiness margin) keeps the zone stable near boundaries,
  so it's settled by the time you press the trigger. Tune with `[` / `]`.
- To swap the trigger to a real **foot pedal** later, just map the pedal to
  ⌘⌥⌃G (or any key) and you're done — no code change.
- Want focus to follow automatically without a trigger? That's the thrashy mode
  we deliberately avoided, but it's a few lines if you change your mind.
