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
	self.discount = args.discount
	self.minibatchSize = args.minibatchSize
	self.l2 = args.l2 -- L2 cost
	self.learning_rate = args.learning_rate

    self.model = self:getNeuralNetwork():cl()
    self.gameDriver = GameDriver.create()
    self.gameDriver:__init()
	self.replayMemory = ReplayMemory({maxSize = 1000,
									  inputDim = 3*56*64,
									  batchSize = 100})
	self.max_reward = 10000
	self.min_reward = -self.max_reward
	self.win_bonus = 3000
	self.time_penalty = 5

	-- auxiliary variables
	selfw, self.dw = self.network:getParameters()
	self.deltas = self.dw.clone():fill(0)
	self.tmp = self.dw:clone():fill(0)
	self.g = self.dw:clone():fill(0)
	self.g2 = self.dw:clone():fill(0)
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
    for i = 1, epochs do
        -- Loading initial save state
        self.gameDriver:loadSaveState()
        -- Getting initial state at time 1
        local start_state = self.gameDriver:getState(t)
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
			local reward = self.countReward(start_state, next_state)
            -- Save in replay memory the transition
			self.replayMemory:add({start_state=start_state, 
                action=action, reward=reward, next_state=next_state})
            -- Now we create random minibatch of transitions from replayMemory
            self:createRandomMiniBatch()
            -- And then we learn the minibatch by performing gradient descent
            self:qLearnMiniBatch()
			start_state = next_state
        end
        -- At the end of an epoch we save our network
        self:saveNeuralNetwork()
    end
end

-- Forwards the input through neural network to update Q values
function DeepQNN:updateQValues()
	local s, a, r, t, s2, delta
	local q, q2, q2_max

	s = self.minibatch.s
	a = self.minibatch.a
	r = self.minibatch.r
	t = self.minibatch.t
	s2 = self.minibatch.s2

	-- t[i] := s2[i] is terminate
	t = t:clone():float():mul(-1):add(1)

	-- compute max_a Q(s2, a) for each state in s2
	q2_max = self.model:forward(s2):float():max(2)

	-- compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
	q2 = a2_max:clone():mul(self.discount):cmul(term)

	-- compute delta = r + (1-terminal) * gamma * max_a Q(s2, a)
	delta = r:clone():float()
	delta:add(q2)

	-- compute q = Q(s,a)
	local q_all = self.model:forward(s):float()
	q = torch.FloatTensor(q_all:size(1))
	for i = 1,q_all:size(1) do
		q[i] = q_all[i][a[i]]
	end
	delta:add(-1, q)

	local targets = torch.zeros(self.minibatchSize, #classes):float()
	for i=1,math.min(self.minibatch_size,a:size(1)) do
		targets[i][a[i]] = delta[i]
	end

	-- store computed data
	self.targets = targets
	self.delta = delta
	self.q2_max = q2_max
end

-- Returns an action based on the epsilon greedy alg
function DeepQNN:eGreedyLearn()
	-- update state
	local input = image.scale(state.screenTensor, dimensions[2], 
		dimensions[3])
	input = (input:view(
		dimensions[1] * dimensions[2] * dimensions[3])):cl()
	state.screenTensor = input

	if torch.uniform() < self.epsilon then
		return self.getAction(torch.random(#self.classes))
	else
		return self.greedyLearn()
	end
end

-- Returns an action based on the greedy alg
function DeepQNN:greedyLearn()
	local last_entry = self.replayMemory:lastTransition()
	local output = self.model:forward(last_entry.next_state)
	return self.getAction(output)
end

-- Learn from previous transitions
function DeepQNN:qLearnMiniBatch()
	local s, a, r, s2, t

	s = self.minibatch.s
	a = self.minibatch.a
	r = self.minibatch.r
	t = self.minibatch.t
	s2 = self.minibatch.s2

	local targets, delta, q2_max = self.updateQValues()

	-- zero gradients parameters
	self.dw:zero()

	-- get new gradient
	self.model:backward(s, targest)

	-- add weight cost to gradient
	self.dw:add(-self.l2, self.w)

	-- use gradients
	self.g:mul(0.95):add(0.05, self.dw)
	self.tmp:dmul(self.dw, self.dw)
	self.g2:mul(0.95):add(0.05, self.tmp)
	self.tmp:cmul(self.g, self.g)
	self.tmp:mul(-1)
	self.tmp:add(self.g2)
	aself.tmp:add(0.01)
	self.tmp:sqrt()

	-- accumulate update
	self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
	self.w:add(self.deltas)
end

function DeepQNN:saveNeuralNetwork()
end

function DeepQNN:createRandomMiniBatch()
	local a, r, s, s2, t = self.replayMemory.sample()
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
			reward = reward + self.win_bonus
		end
	end

	local delta_score = nextState.score - startState.score
	reward = delta_score + self.time_pentaly

	reward = math.min(self.max_reward, reward)
	reward = math.max(self.min_reward, reward)

	return reward / self.max_reward
end

net = DeepQNN.create()
net:__init()
net:test(10000)
