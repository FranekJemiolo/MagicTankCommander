-- This program was written by Franciszek Jemioło
-- In this file we declare all the functions to handle the game environment
-- i.e. reading the score/input/saves
require('gd')
require('torch')
require('cltorch')
require('image')
-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')
GameDriver = {}

GameDriver.__index = GameDriver

-- Constructor of GameDriver
function GameDriver.create()
    local gameDr = {}
    setmetatable(gameDr, GameDriver)
    return gameDr
end

function GameDriver:__init()
    -- The number of the slot used in emulator
    self.predefinedSaveSlot = 1
    -- Screenshot base name
    self.screenshotName = "screen"
    self.screenshotType = ".png"
end


-- Checks how many life we still have (>0 at level won then we win)
function GameDriver:getLives()
    local livesByte = memory.readbyte(0x003B)
    local livesString = string.format("%02X", livesByte)
    livesString = string.format("%u", livesString)
    return tonumber(livesString)
end

-- Checks if we killed all tanks
function GameDriver:isLevelWon()
    -- Always 20 tanks spawned in a level
    local tanksBytes = {}
    -- Read how many tanks did we kill of 4 types each.
    tanksBytes[1] = memory.readbyte(0x0073)
    tanksBytes[2] = memory.readbyte(0x0074)
    tanksBytes[3] = memory.readbyte(0x0075)
    tanksBytes[4] = memory.readbyte(0x0076)
    local tanksStrings = {}
    for i=1,4 do
        tanksStrings[i] = string.format("%02X", tanksBytes[i])
    end
    return (tonumber(tanksStrings[1]) + tonumber(tanksStrings[2]) + 
        tonumber(tanksStrings[3]) + tonumber(tanksStrings[4]) >= 20)
end

-- Checks if game over happend
function GameDriver:isGameOver()
    -- Read from the memory has the game over happend?
    local gameOverByte = memory.readbyte(0x0108)
    -- Remove leading zeroes
    gameOverByte = string.format("%u", gameOverByte)
    return (tonumber(gameOverByte) ~= 0)
end

-- Returns the user score.
function GameDriver:getScore()
    -- Reading 4 bytes of memory where the score is kept
    local scoreBytes = {}
    scoreBytes[1] = memory.readbyte(0x0017)
    scoreBytes[2] = memory.readbyte(0x0018)
    scoreBytes[3] = memory.readbyte(0x0019)
    scoreBytes[4] = memory.readbyte(0x001A)
    -- Displaying as it would be hex value
    local scoreString = string.format("%02X%02X%02X%02X", scoreBytes[1], 
        scoreBytes[2],scoreBytes[3], scoreBytes[4])
    -- Now removing leading zeroes
    scoreString = string.format("%u", scoreString)
    -- Converting from string
    return tonumber(scoreString) 
end

-- This function returns current state in which the game is.
-- Returns the rawimage input, current score, is the game over or level has 
-- ended. Save a screenshot with the "step" number concatenated.
function GameDriver:getState(step)
    -- Creating the gd image object from the emulator api.
    local gdScreenshot = gd.createFromGdStr(gui.gdscreenshot())
    -- Saving the screenshot as png
    gdScreenshot:png(self.screenshotName .. step .. self.screenshotType)
    -- Loading raw png as torch tensor
    local screenTensor = image.load(self.screenshotName .. step .. 
        self.screenshotType, 3, 'float')
    -- Reading the current score
    local score = self:getScore()
    -- Is the state terminal? Meaning do we killed every enemy or we reached
    -- gameover?
    local terminal = self:isLevelWon() or self:isGameOver()
    return {screenTensor=screenTensor, score=score, terminal=terminal}
end

-- Removes all the screenshot created from start to stop
function GameDriver:cleanPreviousScreens(start, stop)
    os.execute("rm -f /home/franek/JNP3/".. self.screenshotName .. "{" .. 
        start .. ".." .. stop .. "}" .. self.screenshotType)
end



-- Loads given save or the initial from predefined slot.
function GameDriver:loadSaveState(save)
    if save == nil then
        save = savestate.object(self.predefinedSaveSlot)
    end
    savestate.load(save)
end

-- Saves in the given slot (1--9)
function GameDriver:saveCurrentState(slot)
    if slot == nil then
        save = savestate.object(self.predefinedSaveSlot)
    else
        save = savestate.object(slot)
    end
    savestate.save(save)
    savestate.persist(save)
end

-- Advancing to the next frame and increasing counter
function GameDriver:advanceToNextFrame(step)
    emu.frameadvance()
    return step + 1
end

-- This function send information to emulator that certain keys are pressed
function GameDriver:sendButtons(input)
    joypad.set(1, input)
end

-- Sets emulator speed mode to maximum
function GameDriver:setMaxSpeed()
    emu.speedmode("maximum")
end

-- Starts new game, returns step counter
function GameDriver:newGame()
    self:loadSaveState()
    return 1
end


--k=1
--while true do
    -- Getting the state
--    state = gameDriver:getState(k)
--    print(string.format("%d", state.score))
--    print(state.terminal)
    -- Reducing dimensions by 4 in width and height
--    local tens = image.scale(state.screenTensor, 56, 64)
--    image.save("1" .. screenshotName .. k .. screenshotType,tens)
    -- Cleaning up previous step
--    if (k % 1000 == 0) then
--        gameDriver:cleanPreviousScreens(k-999, k)
--        print(cleaning)
--    end
--    k = k + 1
--    print(string.format("%d", k))
--    emu.frameadvance()
--end