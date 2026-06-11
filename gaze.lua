-- gaze-focus: commit side of the head-pose window switcher.
-- The Python service keeps ~/.gaze/state.json updated with the current zone:
--   0 = LAPTOP   1 = TOP monitor   2 = RIGHT monitor
-- and, when a screen's split is eye-calibrated, a continuous "gaze_y" in [0,1]
-- (0 = top, 1 = bottom) for where your eyes are within that screen.
-- On a trigger this focuses the matching physical screen -- and if gaze_y is
-- present and the screen holds 2+ windows, the exact window at that height.
--
-- It also draws a live amber "candidate" outline around the window your gaze is
-- on whenever that window ISN'T already focused -- so you can see what mouse4
-- will switch to before you commit. (The commit itself flashes bright cyan.)
--
-- Triggers wired up below:
--   * a keyboard key  (default: F13 / Print Screen -- see TRIGGER_KEY)
--   * hotkey  cmd+alt+ctrl+G  -- commit / focus the candidate (backup)
--   * hotkey  cmd+alt+ctrl+P  -- toggle the live candidate preview
--   * hotkey  cmd+alt+ctrl+K  -- toggle the key detector (find a key's name)
-- Hands-free dwell auto-focus (commit by just looking, no key) is opt-in: start
-- the Python service with --auto-focus. We read that from state.json (below).
--
-- Install: put this file at ~/.hammerspoon/gaze.lua and add this line to
-- ~/.hammerspoon/init.lua:   require("gaze")
-- then click the Hammerspoon menubar icon -> Reload Config.

local M = {}

local STATE_PATH = os.getenv("HOME") .. "/.gaze/state.json"
local FLASH_SECONDS = 0.25
local STALE_SECONDS = 3.0 -- ignore state.json if the service isn't updating it

-- Live candidate preview (the outline that follows your gaze before you commit).
local PREVIEW_ENABLED = true
local PREVIEW_HZ = 10 -- how often to recheck the candidate window
local FLASH_COLOR = { red = 0.23, green = 0.86, blue = 1.0, alpha = 0.95 } -- commit
local PREVIEW_COLOR = { red = 1.0, green = 0.75, blue = 0.15, alpha = 0.95 } -- candidate

-- Optional hard override: map a zone to a screen by (partial) name. Leave a
-- value as nil to auto-detect. Find names with `hs.inspect(hs.screen.allScreens())`
-- in the Hammerspoon console, or System Settings > Displays.
local OVERRIDE = {
  [0] = nil, -- LAPTOP, e.g. "Built-in"
  [1] = nil, -- TOP
  [2] = nil, -- RIGHT
}

local function centerX(s) local f = s:frame() return f.x + f.w / 2 end
local function centerY(s) local f = s:frame() return f.y + f.h / 2 end

-- Resolve zone -> screen for the laptop/top/right layout:
--   RIGHT  = the screen furthest to the right.
--   Of the remaining two: TOP = the higher one (smaller y), LAPTOP = the lower.
-- (macOS y grows downward, so a monitor mounted above the laptop has a smaller y.)
local function resolveScreens()
  local all = hs.screen.allScreens()
  local map = {}

  -- apply any name overrides first
  local used = {}
  for zone = 0, 2 do
    if OVERRIDE[zone] then
      local s = hs.screen.find(OVERRIDE[zone])
      if s then map[zone] = s; used[s:id()] = true end
    end
  end

  local rest = {}
  for _, s in ipairs(all) do
    if not used[s:id()] then rest[#rest + 1] = s end
  end

  if map[2] == nil and #rest > 0 then -- RIGHT = furthest right
    local right = rest[1]
    for _, s in ipairs(rest) do
      if centerX(s) > centerX(right) then right = s end
    end
    map[2] = right
  end

  local remaining = {}
  for _, s in ipairs(rest) do
    if not (map[2] and s:id() == map[2]:id()) then remaining[#remaining + 1] = s end
  end

  if #remaining >= 2 then
    local a, b = remaining[1], remaining[2]
    local top, laptop
    if centerY(a) < centerY(b) then top, laptop = a, b else top, laptop = b, a end
    if map[1] == nil then map[1] = top end
    if map[0] == nil then map[0] = laptop end
  elseif #remaining == 1 then
    if map[1] == nil then map[1] = remaining[1] end
    if map[0] == nil then map[0] = remaining[1] end
  end

  -- final fallback: primary screen for anything still unset
  local primary = hs.screen.primaryScreen()
  for zone = 0, 2 do
    if map[zone] == nil then map[zone] = primary end
  end
  return map
end

local function readState()
  local f = io.open(STATE_PATH, "r")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  local zone = content:match('"zone"%s*:%s*(%-?%d+)')
  local ts = content:match('"ts"%s*:%s*([%d%.]+)')
  -- gaze_y is null when the active screen has no eye calibration -> no match.
  local gy = content:match('"gaze_y"%s*:%s*(%-?[%d%.]+)')
  -- Hands-free dwell focus, set by the service's --auto-focus flag (absent on
  -- older builds -> treated as off).
  local af = content:match('"auto_focus"%s*:%s*(%a+)')
  local afs = content:match('"auto_focus_seconds"%s*:%s*([%d%.]+)')
  if not zone then return nil end
  return tonumber(zone), tonumber(ts), tonumber(gy), af == "true", tonumber(afs)
end

-- Flash a cyan outline around any rect (a whole screen, or a single window).
local function flashFrame(rect)
  local c = hs.canvas.new(rect)
  c:appendElements({
    type = "rectangle",
    action = "stroke",
    strokeColor = FLASH_COLOR,
    strokeWidth = 8,
    roundedRectRadii = { xRadius = 10, yRadius = 10 },
    frame = { x = 0, y = 0, w = rect.w, h = rect.h },
  })
  c:canvasMouseEvents(false, false, false, false) -- never eat clicks
  c:show()
  hs.timer.doAfter(FLASH_SECONDS, function() c:delete() end)
end

-- Pick the window on `target` whose vertical position best matches gazeY.
-- gazeY in [0,1]: 0 = topmost window, 1 = bottommost. Returns nil if there
-- aren't enough windows or no gaze signal -- caller falls back to frontmost.
local function windowAtGaze(wins, gazeY)
  if gazeY == nil or #wins < 2 then return nil end
  local minC, maxC = math.huge, -math.huge
  for _, w in ipairs(wins) do
    local f = w:frame()
    local c = f.y + f.h / 2
    if c < minC then minC = c end
    if c > maxC then maxC = c end
  end
  local span = maxC - minC
  if span < 1 then return nil end -- windows stacked, can't separate vertically
  local best, bestErr = nil, math.huge
  for _, w in ipairs(wins) do
    local f = w:frame()
    local norm = (f.y + f.h / 2 - minC) / span
    local err = math.abs(norm - gazeY)
    if err < bestErr then bestErr = err; best = w end
  end
  return best
end

-- visible standard windows on a screen, front-to-back
local function windowsOnScreen(target)
  local wins = {}
  for _, w in ipairs(hs.window.orderedWindows()) do
    if w:screen():id() == target:id() and w:isStandard() then
      wins[#wins + 1] = w
    end
  end
  return wins
end

-- The window a commit would focus right now. Returns (window, targetScreen);
-- window is nil if the target screen has no windows.
local function candidate(zone, gazeY)
  if not zone or zone < 0 or zone > 2 then return nil, nil end
  local target = resolveScreens()[zone]
  if not target then return nil, nil end
  local wins = windowsOnScreen(target)
  if #wins == 0 then return nil, target end
  return (windowAtGaze(wins, gazeY) or wins[1]), target -- gaze, else frontmost
end

-- Persistent amber outline around the current candidate window.
local previewCanvas, previewSig = nil, nil

local function hidePreview()
  if previewCanvas then previewCanvas:delete(); previewCanvas = nil end
  previewSig = nil
end

local function showPreview(win)
  local f = win:frame()
  local sig = string.format("%d:%.0f:%.0f:%.0f:%.0f", win:id(), f.x, f.y, f.w, f.h)
  if sig == previewSig then return end -- already outlining this exact frame
  hidePreview()
  local c = hs.canvas.new(f)
  c:appendElements({
    type = "rectangle",
    action = "stroke",
    strokeColor = PREVIEW_COLOR,
    strokeWidth = 6,
    roundedRectRadii = { xRadius = 10, yRadius = 10 },
    frame = { x = 0, y = 0, w = f.w, h = f.h },
  })
  c:level(hs.canvas.windowLevels.floating)
  c:canvasMouseEvents(false, false, false, false) -- click-through
  c:clickActivating(false)
  c:show()
  previewCanvas, previewSig = c, sig
end

local function focusZone(zone, gazeY)
  local pick, target = candidate(zone, gazeY)
  if not target then return end
  hidePreview()
  if not pick then
    flashFrame(target:fullFrame()) -- nothing there; still flash for feedback
    return
  end
  pick:focus()
  flashFrame(pick:frame())
end

local function commit()
  local zone, ts, gy = readState()
  if zone == nil then return end
  if ts and (os.time() - ts) > STALE_SECONDS then
    hs.alert.show("gaze: pose service not running?")
    return
  end
  focusZone(zone, gy)
end

-- ── Dwell auto-focus ────────────────────────────────────────────────────────
-- Optional hands-free mode: instead of tapping the trigger, just keep looking.
-- If the candidate window stays the same (and isn't already focused) for the
-- dwell time, focus commits on its own. This is driven entirely by the Python
-- service: run it with `--auto-focus` (optionally `--auto-focus-seconds N`) and
-- it writes auto_focus / auto_focus_seconds into state.json, which we read each
-- tick. No flag -> auto-focus stays off and only the manual trigger commits.
local AUTO_FOCUS_DEFAULT_SECONDS = 3.0 -- fallback if state.json omits the seconds

local dwellWinId, dwellStart = nil, nil -- which candidate we're dwelling on, since when
local function resetDwell() dwellWinId, dwellStart = nil, nil end

-- Per-tick: keep the live candidate preview in sync AND, when the service has
-- auto-focus on, track how long you've held the same unfocused candidate and
-- commit once it crosses the dwell threshold. One candidate read feeds both.
local function tick()
  local zone, ts, gy, autoFocus, autoSecs = readState()
  if zone == nil or (ts and (os.time() - ts) > STALE_SECONDS) then
    hidePreview(); resetDwell(); return
  end
  local pick = candidate(zone, gy)
  if not pick then hidePreview(); resetDwell(); return end

  local focused = hs.window.focusedWindow()
  local alreadyFocused = focused and pick:id() == focused:id()

  -- Live preview: outline the candidate unless it's already where focus is.
  if PREVIEW_ENABLED and not alreadyFocused then showPreview(pick) else hidePreview() end

  -- Dwell only counts on an unfocused candidate, and only when --auto-focus is on.
  if not autoFocus or alreadyFocused then resetDwell(); return end
  local now = hs.timer.secondsSinceEpoch()
  if pick:id() ~= dwellWinId then
    dwellWinId, dwellStart = pick:id(), now -- candidate changed; restart the clock
  elseif now - dwellStart >= (autoSecs or AUTO_FOCUS_DEFAULT_SECONDS) then
    focusZone(zone, gy) -- held long enough -- commit, then wait for the next look
    resetDwell()
  end
end

M.previewTimer = hs.timer.doEvery(1.0 / PREVIEW_HZ, tick)

-- Toggle the live candidate preview on/off.
hs.hotkey.bind({ "cmd", "alt", "ctrl" }, "p", function()
  PREVIEW_ENABLED = not PREVIEW_ENABLED
  if not PREVIEW_ENABLED then hidePreview() end
  hs.alert.show("gaze preview: " .. (PREVIEW_ENABLED and "on" or "off"))
end)

-- ── Commit trigger ──────────────────────────────────────────────────────────
-- The key that commits focus to the candidate window. Good "dead" keys on
-- macOS that won't clash with anything: Print Screen, Scroll Lock, Pause. AVOID
-- Home / End / Page Up / Page Down -- you use those constantly for scrolling and
-- cursor movement, so binding one would break normal typing.
--
-- TRIGGER_KEY takes a Hammerspoon key name (e.g. "f13", "pause", "f14") OR a raw
-- keycode number (e.g. 113). Most PC keyboards on a Mac send F13/F14/F15 from the
-- Print Screen / Scroll Lock / Pause cluster -- so "f13" is the default.
--
-- Not sure what your key sends? Press cmd+alt+ctrl+K to turn on the detector, tap
-- your key, and an alert (+ the Hammerspoon console) shows its name and keycode.
-- Put that here: the name if one is shown, otherwise the raw number.
local TRIGGER_KEY = "f13"
local TRIGGER_MODS = {} -- e.g. { "cmd", "alt" } for a chord; {} for a bare key

local triggerHotkey = hs.hotkey.bind(TRIGGER_MODS, TRIGGER_KEY, commit)
if not triggerHotkey then
  hs.alert.show("gaze: trigger key '" .. tostring(TRIGGER_KEY) ..
    "' not recognized -- use the detector (cmd+alt+ctrl+K)")
end

-- Backup hotkey, always available regardless of TRIGGER_KEY.
hs.hotkey.bind({ "cmd", "alt", "ctrl" }, "g", commit)

-- Key detector: toggle with cmd+alt+ctrl+K, then press a key to see its name +
-- keycode so you know what to set TRIGGER_KEY to. Off by default; it never
-- swallows keys, so normal typing is unaffected while it's on.
local keyDetector = nil
hs.hotkey.bind({ "cmd", "alt", "ctrl" }, "k", function()
  if keyDetector then
    keyDetector:stop(); keyDetector = nil
    hs.alert.show("gaze key detector: off")
    return
  end
  keyDetector = hs.eventtap.new({ hs.eventtap.event.types.keyDown }, function(e)
    local code = e:getKeyCode()
    local name = hs.keycodes.map[code]
    local label = name and ('"' .. name .. '"') or ("keycode " .. code)
    print(string.format("[gaze] key pressed: name=%s keycode=%d", tostring(name), code))
    hs.alert.show("gaze: this key is " .. label)
    return false -- never swallow the key
  end)
  keyDetector:start()
  hs.alert.show("gaze key detector: ON -- press your key")
end)

hs.alert.show("gaze-focus loaded")

return M
