-- This program was written by Franciszek Jemioło
-- In this file I declare the model of the network.
require('torch')
require('cltorch')
require('nn')
require('clnn')
require('image')
require('gamedriver')
require('replay_memory')
-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')
-- The dimensions of the images percieved by our network
dimensions = {3, 56, 64}
-- 6 classes because we have 6 action buttons
classes = {1, 2, 3, 4, 5, 6}

-- Creating our class
DeepQNN = {}

DeepQNN.__index = DeepQNN


function DeepQNN.create()
    local dpqnn = {}
    setmetatable(dpqnn, DeepQNN)
    return dpqnn
end

function DeepQNN:__init(args)
    self.model = self:getNeuralNetwork():cl()
    self.gameDriver = GameDriver.create()
    self.gameDriver:__init()
	self.discount = args.discount
	self.replayMemory = ReplayMemory
	self.replayCounter = 1
	self.minibatchSize = args.minibatchSize
end


-- This function returns the model of our neural network
function DeepQNN:getNeuralNetwork(hidden)
    local hus = hidden or 4096
    local net = nn.Sequential()
    net:add(nn.Linear(dimensions[1] * dimensions[2] * dimensions[3], hus))
    net:add(nn.Tanh())
    net:add(nn.Linear(hus, #classes))
    net:add(nn.LogSoftMax())
    return net
end

-- Returns chosen action by the neural network
function DeepQNN:getAction(output)
    print(output)
    local max = output[1]
    local maxi = 1
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
function DeepQNN:test(steps)
    local endOfGame = false
    local i = 1
    -- Load inital save
    self.gameDriver:loadSaveState()
    while ((i <= steps) and (not endOfGame)) do
        -- Getting current state
        local state = self.gameDriver:getState(i)
        endOfGame = state.terminal
        if (not endOfGame) then
            -- Reshaping our vector so it fits our network
            local input = image.scale(state.screenTensor, dimensions[2], 
                dimensions[3])
            input = (input:view(
                dimensions[1] * dimensions[2] * dimensions[3])):cl()
            local output = self.model:forward(input)
            -- Getting set of keys to press
            local action = self:getAction(output)
            -- Sending the keys
            self.gameDriver:sendButtons(action)
            -- Playing next frame
            i = self.gameDriver:advanceToNextFrame(i)
            -- Now we should evaluate our action
        else 
            -- Restarting game
            endOfGame = false
            self.gameDriver:loadSaveState()
        end
    end
end

function DeepQNN:qLearnMiniBatch()
	self.miniBatch = {}
	local y = nil
	local w = nil
	local x = nil
	local gradient = 0

	for i = 1,#self.miniBatch do
		x = self.miniBatch[i]
		local output = self.model:forward(x.next_state)
		if x.next_state.terminal then
			y = x.reward
		else
			local max = output[1]
			for i = 1,#classes do
				if (output[i] > max) then
					max = output[i]
				end
			end
			y = x.reward + self.discount * max
		end
		w = output[x.action]
		-- gradient = gradient + (y - w)*
		-- compute component of subtotal gradient

	end

end

-- In this function we train our network based on the Q-Learning algorithm
-- The steps count is for how many frames will it play
function DeepQNN:train(epochs, steps)
    -- Perform #epochs when training, at the end of every epoch save our network
    for i = 1, epochs do
        -- Loading initial save state
        self.gameDriver:loadSaveState()
        -- Getting initial state at time 1
        local start_state = self.gameDriver:getState(t)
        -- Forward the state through the network to get the q value
        self:updateQValues()
        local t = 1
        while (t <= steps) do
            -- Choosing action based on eGreedy alg/ here we calculate reward
            local action = self:eGreedyLearn()
            -- Executing action
            self.gameDriver:sendButtons(action)
            -- Get the next frame
            t = self.gameDriver:advanceToNextFrame(t)
            -- Get the state at time t + 1
            local next_state = self.gameDriver:getState(t)
            -- Save in replay memory the transition
			self.replayMemory:add({start_state=start_state, 
                action=action, reward=reward, next_state=next_state})
            -- Now we create random minibatch of transitions from replayMEmory
            self:createRandomMiniBatch()
            -- And then we learn the minibatch by performing gradient descent
            self:qLearnMiniBatch()
        end
        -- At the end of an epoch we save our network
        self:saveNeuralNetwork()
    end
end

-- Forwards the input through neural network to update Q values
function DeepQNN:updateQValues()
end

-- Returns an action based on the epsilon greedy alg
function DeepQNN:eGreedyLearn()
end

-- Returns an action based on the greedy alg
function DeepQNN:greedyLearn()
	if torch.uniform() < self.epsilon then
		return self.getAction(torch.random(#self.classes))
	else
		local last_entry = self.replayMemory:lastTransition()
		local output = self.model:forward(last_entry.next_state)
		return self.getAction(output)
	end
end

-- Learn from previous transitions
function DeepQNN:qLearnMiniBatch()
end

function DeepQNN:saveNeuralNetwork()
end

function DeepQNN:createRandomMiniBatch()
	self.minibatch = self.replayMemory.sample(self.minibatchSize)
end

-- Returns reward for the choosen action
function DeepQNN:countReward()
	local won_level_reward = 1000
	local game_over_reward = -1000
	local time_pentaly = -1
	local reward = 0

	local last_entry = self.replayMemory:lastTransition()
	local delta_score = last_entry.next_state - last_entry.start_state

	reward = delta_score + time_pentaly
	if last_entry.next_state.terminal then
		if last_entry.next_state.isLevelWon then
			reward = reward + won_level_reward
		else
			reward = reward + game_over_reward
		end
	end

	return reward
end


net = DeepQNN.create()
net:__init()
net:test(10000)
