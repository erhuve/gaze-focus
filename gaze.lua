-- gaze-focus: commit side of the head-pose window switcher.
-- The Python service keeps ~/.gaze/state.json updated with the current zone
-- (0=left, 1=center, 2=right). This reads that zone on a trigger and focuses
-- the frontmost window on the matching monitor.
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

-- Sort screens left-to-right by their x position so zone 0/1/2 maps to
-- leftmost/middle/rightmost regardless of how macOS orders them.
local function screensLeftToRight()
  local screens = hs.screen.allScreens()
  table.sort(screens, function(a, b)
    return a:frame().x < b:frame().x
  end)
  return screens
end

local function readZone()
  local f = io.open(STATE_PATH, "r")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  local zone = content:match('"zone"%s*:%s*(%-?%d+)')
  if zone then return tonumber(zone) end
  return nil
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
  if not zone then return end
  local screens = screensLeftToRight()
  local target = screens[zone + 1] -- zone is 0-indexed
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
  focusZone(readZone())
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
