-- This program was written by Franciszek Jemioło
-- In this file I declare the model of the network.
require('torch')
require('cltorch')
require('nn')
require('clnn')
require('image')
require('gamedriver')
require('replaymemory')
require('xlua')
-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')

-- Creating our class
DeepQNN = {}

DeepQNN.__index = DeepQNN

function DeepQNN.create()
    local dpqnn = {}
    setmetatable(dpqnn, DeepQNN)
    return dpqnn
end

function DeepQNN:__init(args)
    -- The dimensions of the images percieved by our network
    self.dimensions = {3, 56, 64}
    -- 6 classes because we have 6 action buttons
    self.classes = {1, 2, 3, 4, 5, 6}

    -- On how many epochs the net was trained
    self.passedEpochs = 0
    self.modelFilename = "model"
    -- Parametrs of Q function
	self.discount = args.discount or 0.3
	self.minibatchSize = args.minibatchSize or 2
    -- How many frames must pass for us to learn again qLearnMiniBatch
    self.minibatchLearnRate = 10
    self.minibatchCounter = 0
	self.l2 = args.l2 or 0.01-- L2 cost 
	self.learning_rate = args.learning_rate or 0.1
    self.epsilon = 0.1

    -- Creating model of neural network
    if args.load == true then
        self:loadNeuralNetwork()
    else
        self.model = self:getNeuralNetwork():cl()
        local replayMemoryArgs = {maxSize = 100000,
        inputDim = self.dimensions[1] * self.dimensions[2] * self.dimensions[3],
        batchSize = self.minibatchSize}
        -- Creating replay memory to store there transitions between states
        self.replayMemory = ReplayMemory.create()

        self.replayMemory:__init(replayMemoryArgs)
    end
    -- Creating our game driver to handle emulator events
    self.gameDriver = GameDriver.create()
    self.gameDriver:__init()
    self.gameDriver:setMaxSpeed()

    -- Max rewards
	self.maxReward = 10000
	self.minReward = -self.maxReward
    -- Bonus points and time penalty
	self.winBonus = 3000
	self.timePenalty = 1
    -- Reward range (-a, a), used for scaling reward
    self.rewardRange = 100

	-- parameters, gradParameters of our model
	self.w, self.dw = self.model:getParameters()
    -- Auxiliary parameters
	self.deltas = self.dw:clone():fill(0)
	self.tmp = self.dw:clone():fill(0)
	self.g = self.dw:clone():fill(0)
	self.g2 = self.dw:clone():fill(0)
end


-- This function returns the model of our neural network
function DeepQNN:getNeuralNetwork(hidden)
    local hus = hidden or 4096
    local net = nn.Sequential()
    net:add(nn.Linear(self.dimensions[1] * 
        self.dimensions[2] * self.dimensions[3], hus))
    net:add(nn.Tanh())
    net:add(nn.Linear(hus, #self.classes))
    --net:add(nn.LogSoftMax())
    return net
end

-- Returns chosen action by the neural network
function DeepQNN:getAction(output)
    local max = output[1]
    local maxi = 1
    for i = 1, #self.classes do
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

function DeepQNN:actionToNumber(action)
    if action['up'] == true then
        return 1
    elseif action['down'] == true then
        return 2
    elseif action['left'] == true then
        return 3
    elseif action['right'] == true then
        return 4
    elseif action['A'] == true then
        return 5
    elseif action['B'] == true then
        return 6
    end
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
        xlua.progress(i, steps)
        local state = self.gameDriver:getState(i)
        endOfGame = state.terminal
        if (not endOfGame) then
            -- Reshaping our vector so it fits our network
            --local input = image.scale(state.screenTensor, dimensions[2], 
            --    dimensions[3])
            input = state.screenTensor
            input = (input:view(self.dimensions[1] * self.dimensions[2] * 
                self.dimensions[3])):cl()
			-- update state
			state.screenTensor = input
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

-- In this function we train our network based on the Q-Learning algorithm
-- The steps count is for how many frames will it play
function DeepQNN:train(epochs, steps)
    -- Perform #epochs when training, at the end of every epoch save our network
    local start_state
    local next_state
    local reward
    for i = 1, epochs do
        -- Loading initial save state
        self.gameDriver:loadSaveState()
        -- Getting initial state at time 1
        local t = 1
        start_state = self.gameDriver:getState(t)
        print("Starting epoch :" .. i)
        -- Terminating epoch training if level has ended or game over happend
        while (t <= steps) do--and (not start_state.terminal) do
            xlua.progress(t, steps)
            if start_state.terminal then
                -- Restarting game
                self.gameDriver:loadSaveState()
                start_state = self.gameDriver:getState(t)
            end
            -- Choosing action based on eGreedy alg/ here we calculate reward
            local action = self:eGreedyLearn(start_state)
            -- Executing action
            self.gameDriver:sendButtons(action)
            -- Get the next frame
            t = self.gameDriver:advanceToNextFrame(t)
            -- Get the state at time t + 1
            next_state = self.gameDriver:getState(t)
			reward = self:countReward(start_state, next_state)
            -- Save in replay memory the transition
			self.replayMemory:add({start_state=start_state, 
                action=self:actionToNumber(action), reward=reward, 
                next_state=next_state})
            if (self.minibatchCounter == self.minibatchLearnRate) then
                -- Now we create random minibatch of transitions from 
                -- replayMemory
                self:createRandomMiniBatch()
                -- And then we learn the minibatch by performing gradient 
                -- descent
                self:qLearnMiniBatch()
                self.minibatchCounter = 0
            end
            self.minibatchCounter = self.minibatchCounter + 1
			start_state = next_state
        end
        self.passedEpochs = self.passedEpochs + 1
        -- At the end of an epoch we save our network
        print("Passed epoch : " .. self.passedEpochs)
        self:saveNeuralNetwork()
        collectgarbage()
    end
end

-- Forwards the input through neural network to update Q values
function DeepQNN:updateQValues()
	local s, a, r, t, s2, delta
	local q, q2, q2Max

	s = self.minibatch.s
	a = self.minibatch.a
	r = self.minibatch.r
	t = self.minibatch.t
	s2 = self.minibatch.s2

	-- t[i] := s2[i] is terminate
	t = t:mul(-1):add(1)

	-- compute max_a Q(s2, a) for each state in s2
	q2Max = self.model:forward(s2:view(self.minibatchSize, self.dimensions[1] * 
        self.dimensions[2] * self.dimensions[3]):cl()):max(2)

	-- compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
	q2 = q2Max:clone():mul(self.discount):cmul(t:cl())

	-- compute delta = r + (1-terminal) * gamma * max_a Q(s2, a)
	delta = r:clone():cl()
	delta:add(q2):cl()

	-- compute q = Q(s,a)
	local q_all = self.model:forward(s:view(self.minibatchSize, 
        self.dimensions[1] * self.dimensions[2] * 
        self.dimensions[3]):cl())
	q = torch.ClTensor(q_all:size(1))
	for i = 1,q_all:size(1) do
		q[i] = q_all[i][a[i]]
	end
	delta:add(-1, q)

	local targets = torch.zeros(self.minibatchSize, #self.classes):cl()
	for i=1,math.min(self.minibatchSize,a:size(1)) do
		targets[i][a[i]] = delta[i]
	end

	-- store computed data
	self.targets = targets
	self.delta = delta
	self.q2Max = q2Max
    return targets, delta, q2Max
end

-- Returns an action based on the epsilon greedy alg
function DeepQNN:eGreedyLearn(state)
	if torch.uniform() < self.epsilon then
		return self:getAction(torch.rand(#self.classes))
	else
		return self:greedyLearn(state)
	end
end

-- Returns an action based on the greedy alg
function DeepQNN:greedyLearn(state)
	local output = self.model:forward(state.screenTensor:view(
        self.dimensions[1] * self.dimensions[2] * self.dimensions[3]):cl())
	return self:getAction(output)
end

-- Learn from previous transitions
function DeepQNN:qLearnMiniBatch()
	local s, a, r, s2, t

	s = self.minibatch.s
	a = self.minibatch.a
	r = self.minibatch.r
	t = self.minibatch.t
	s2 = self.minibatch.s2

	local targets, delta, q2Max = self:updateQValues()

	-- Zeroing gradParameters
	self.dw:zero()

	-- Getting new gradient
	self.model:backward(s:cl(), targets)

	-- Adding weight cost to gradient (l2 regularization)
	self.dw:add(-self.l2, self.w)

	-- Using gradients to compute our "criterion"
	self.g:mul(0.95):add(0.05, self.dw)
	self.tmp:cmul(self.dw, self.dw)
	self.g2:mul(0.95):add(0.05, self.tmp)
	self.tmp:cmul(self.g, self.g)
	self.tmp:mul(-1)
	self.tmp:add(self.g2)
	self.tmp:add(0.01)
	self.tmp:sqrt()

	-- Accumulating update
	self.deltas:mul(0):addcdiv(self.learning_rate, self.dw, self.tmp)
	self.w:add(self.deltas)
end

function DeepQNN:saveNeuralNetwork()
    torch.save(self.modelFilename .. ".t7", {
        model = self.model,
        replayMemory = self.replayMemory,
        epochs = self.passedEpochs,
        minibatchSize = self.minibatchSize
    }, "binary", true)
end

function DeepQNN:loadNeuralNetwork()
    local saveModel = torch.load(self.modelFilename .. ".t7", "binary", true)
    self.model = saveModel.model
    self.replayMemory = saveModel.replayMemory
    setmetatable(self.replayMemory, ReplayMemory)
    self.passedEpochs = saveModel.epochs
    self.minibatchSize = saveModel.minibatchSize
end


function DeepQNN:createRandomMiniBatch()
	local a, r, s, s2, t = self.replayMemory:sample()

	self.minibatch = {a=a, r=r, s=s, s2=s2, t=t}
end

-- Returns reward for the choosen action
function DeepQNN:countReward(startState, nextState)
	local reward = 0

	-- when game over return -1
	if nextState.terminal then
		if not nextState.isLevelWon then
			return -1
		else
			reward = reward + self.winBonus
		end
	end

	local delta_score = nextState.score -- - startState.score
	reward = reward + delta_score + self.timePenalty

	reward = math.min(self.maxReward, reward)
	reward = math.max(self.minReward, reward)
    return (reward / self.maxReward) * self.rewardRange
end

net = DeepQNN.create()
args = {load=true} -- {}
net:__init(args)
net:train(1000, 10000)
