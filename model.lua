-- This program was written by Franciszek Jemioło
-- In this file I declare the model of the network.
require('torch')
require('cltorch')
require('nn')
require('clnn')
require('image')
require('gamedriver')
-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')
-- The dimensions of the images percieved by our network
dimensions = {3, 56, 64}
-- 6 classes because we have 6 action buttons
classes = {1, 2, 3, 4, 5, 6}

-- This function returns the model of our neural network
function getNeuralNetwork(hidden)
    local hus = hidden or 4096
    local net = nn.Sequential()
    net:add(nn.Linear(dimensions[1] * dimensions[2] * dimensions[3], hus))
    net:add(nn.Tanh())
    net:add(nn.Linear(hus, #classes))
    net:add(nn.LogSoftMax())
    return net
end

-- Returns chosen action by the neural network
function getAction(output)
    local max = 0
    local maxi = 0
    for i = 1, #classes do
        if (output[i] > max) then
            max = output[i]
            maxi = i
        end
    end
    local input = {}
    input['up'] = false
    input['down'] = false
    input['left'] = false
    input['right'] = false
    input['A'] = false
    input['B'] = false
    input['start'] = false
    input['select'] = false
    if maxi == 0 then
        -- Do nothing/take no action
    elseif maxi == 1 then
        input['up'] = true
    elseif maxi == 2 then
        input['down'] = true
    elseif maxi == 3 then
        input['left'] = true
    elseif maxi == 4 then
        input['right'] = true
    elseif maxi == 5 then
        input['A'] = true
    elseif maxi == 6 then
        input['B'] = true
    end
    return input
end


-- In this function we test our network - how it performs in live action
-- Number of steps is how many frames it will play
function test(neuralNetwork, steps)
    local endOfGame = false
    local i = 1
    -- Load inital save
    gameDriver:loadSaveState()
    while ((i <= steps) and (not endOfGame)) do
        -- Getting current state
        local state = gameDriver:getState(i)
        -- Reshaping our vector so it fits our network
        local input = image.scale(state.screenTensor, dimensions[2], 
            dimensions[3])
        input = (input:view(dimensions[1] * dimensions[2] * dimensions[3])):cl()
        local output = neuralNetwork:forward(input)
        -- Getting set of keys to press
        local action = getAction(output)
        -- Sending the keys
        gameDriver:sendButtons(action)
        -- Playing next frame
        i = gameDriver:advanceToNextFrame(i)
        -- Now we should evaluate our action
    end
end

-- In this function we train our network based on the Q-Learning algorithm
-- The steps count is for how many frames will it play
function train(neuralNetwork, steps)
end

model = getNeuralNetwork():cl()
test(model, 1000)