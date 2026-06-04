-- gaze-focus: commit side of the head-pose window switcher.
-- The Python service keeps ~/.gaze/state.json updated with the current zone:
--   0 = LAPTOP   1 = TOP monitor   2 = RIGHT monitor
-- This reads that zone on a trigger and focuses the frontmost window on the
-- matching physical screen.
--
-- Triggers wired up below:
--   * mouse4 (the "back" thumb button) -- no remapping needed
--   * hotkey  cmd+alt+ctrl+G           -- change to whatever you like
--
-- Install: put this file at ~/.hammerspoon/gaze.lua and add this line to
-- ~/.hammerspoon/init.lua:   require("gaze")
-- then click the Hammerspoon menubar icon -> Reload Config.

local M = {}

local STATE_PATH = os.getenv("HOME") .. "/.gaze/state.json"
local FLASH_SECONDS = 0.25
local STALE_SECONDS = 3.0 -- ignore state.json if the service isn't updating it

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
  if not zone then return nil end
  return tonumber(zone), tonumber(ts)
end

local function flashScreen(screen)
  local frame = screen:fullFrame()
  local c = hs.canvas.new(frame)
  c:appendElements({
    type = "rectangle",
    action = "stroke",
    strokeColor = { red = 0.23, green = 0.86, blue = 1.0, alpha = 0.95 },
    strokeWidth = 10,
    roundedRectRadii = { xRadius = 12, yRadius = 12 },
    frame = { x = 0, y = 0, w = frame.w, h = frame.h },
  })
  c:show()
  hs.timer.doAfter(FLASH_SECONDS, function() c:delete() end)
end

local function focusZone(zone)
  if not zone or zone < 0 or zone > 2 then return end
  local target = resolveScreens()[zone]
  if not target then return end

  -- frontmost standard window on the target screen
  for _, w in ipairs(hs.window.orderedWindows()) do
    if w:screen():id() == target:id() and w:isStandard() then
      w:focus()
      flashScreen(target)
      return
    end
  end
  -- no window there; still flash so you get feedback
  flashScreen(target)
end

local function commit()
  local zone, ts = readState()
  if zone == nil then return end
  if ts and (os.time() - ts) > STALE_SECONDS then
    hs.alert.show("gaze: pose service not running?")
    return
  end
  focusZone(zone)
end

-- Hotkey trigger (change the mods/key to taste)
hs.hotkey.bind({ "cmd", "alt", "ctrl" }, "g", commit)

-- Mouse4 (back button) trigger. buttonNumber 3 = mouse4, 4 = mouse5.
M.mouseTap = hs.eventtap.new({ hs.eventtap.event.types.otherMouseDown }, function(e)
  local btn = e:getProperty(hs.eventtap.event.properties.mouseEventButtonNumber)
  if btn == 3 then
    commit()
  end
  return false -- don't swallow the event
end)
M.mouseTap:start()

hs.alert.show("gaze-focus loaded")

return M
