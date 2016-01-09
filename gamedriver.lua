-- This program was written by Franciszek Jemioło
-- In this file we declare all the functions to handle the game environment
-- i.e. reading the score/input/saves
require('gd')
require('torch')
require('cltorch')
require('image')
-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')
gameDriver = {}

-- The number of the slot used in emulator
predefinedSaveSlot = 1

-- Screenshot base name
screenshotName = "screen"
screenshotType = ".png"

-- Checks how many life we still have (>0 at level won then we win)
function getLives()
    local livesByte = memory.readbyte(0x003B)
    local livesString = string.format("%02X", livesByte)
    livesString = string.format("%u", livesString)
    return tonumber(livesString)
end

-- Checks if we killed all tanks
function isLevelWon()
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
function isGameOver()
    -- Read from the memory has the game over happend?
    local gameOverByte = memory.readbyte(0x0108)
    -- Remove leading zeroes
    gameOverByte = string.format("%u", gameOverByte)
    return (tonumber(gameOverByte) ~= 0)
end

-- Returns the user score.
function getScore()
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
function gameDriver:getState(step)
    -- Creating the gd image object from the emulator api.
    local gdScreenshot = gd.createFromGdStr(gui.gdscreenshot())
    -- Saving the screenshot as png
    gdScreenshot:png(screenshotName .. step .. screenshotType)
    -- Loading raw png as torch tensor
    local screenTensor = image.load(screenshotName .. step .. screenshotType, 3,
        'float')
    -- Reading the current score
    local score = getScore()
    -- Is the state terminal? Meaning do we killed every enemy or we reached
    -- gameover?
    local terminal = isLevelWon() or isGameOver()
    return {screenTensor=screenTensor, score=score, terminal=terminal}
end

-- Removes all the screenshot created from start to stop
function gameDriver:cleanPreviousScreens(start, stop)
    os.execute("rm -f /home/franek/JNP3/".. screenshotName .. "{" .. 
        start .. ".." .. stop .. "}" .. screenshotType)
end



-- Loads given save or the initial from predefined slot.
function gameDriver:loadSaveState(save)
    if save == nil then
        save = savestate.object(predefinedSaveSlot)
    end
    savestate.load(save)
end

-- Saves in the given slot (1--9)
function gameDriver:saveCurrentState(slot)
    if slot == nil then
        save = savestate.object(predefinedSaveSlot)
    else
        save = savestate.object(slot)
    end
    savestate.save(save)
    savestate.persist(save)
end

-- Advancing to the next frame and increasing counter
function gameDriver:advanceToNextFrame(step)
    emu.frameadvance()
    return step + 1
end

-- This function send information to emulator that certain keys are pressed
function gameDriver:sendButtons(input)
    joypad.set(1, input)
end

-- Sets emulator speed mode to maximum
function gameDriver:setMaxSpeed()
    emu.speedmode("maximum")
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