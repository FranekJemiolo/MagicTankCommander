-- This program was written by Franciszek Jemioło
-- In this file I declare the model of the network.
require('torch')
require('nn')
require('cltorch')
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
    self.dimensions = {3, 112, 112}
    -- 7 classes because we have 6 action buttons and last does nothing
    self.classes = {1, 2, 3, 4, 5, 6, 7}

    -- On how many epochs the net was trained
    self.passedEpochs = 0
    self.modelFilename = args.modelFilename or "model"
    self.modelFilenameLoad = args.modelFilenameLoad or "model"
    self.modelBackupFilename = ((args.modelFilename ) or "model") .."-backup"
    -- Parametrs of Q function
	self.discount = args.discount or 0.98
	self.minibatchSize = args.minibatchSize or 32
    -- How many frames must pass for us to learn again qLearnMiniBatch
    self.minibatchLearnRate = 1
    self.replayMemoryMaxSize = 50000
    self.minibatchCounter = 0
	self.l2 = args.l2 or 0.000001-- L2 cost 
	self.learning_rate_start = args.learning_rate or 0.0001
    self.learning_rate = self.learning_rate_start 
    self.learning_rate_end = args.learning_rate_end or 0.0001
    self.learning_rate_endt = args.lr_endt or 2000000
    self.epsilon = 0.5
    --self.levelScale = 0.01
    self.levelScale = 0.1
    self.steps = 0
    --self.stepW = 0.000001
    self.stepW = 0.00001
    self.epsilon_start = 1
    self.epsilon_end = 0.1
    self.lastSavedBackup = false

    -- Creating model of neural network
    if args.load == true then
        self:loadNeuralNetwork()
        local replayMemoryArgs = {maxSize = self.replayMemoryMaxSize,
        inputDim = self.dimensions[1] * self.dimensions[2] * self.dimensions[3],
        batchSize = self.minibatchSize}
        -- Creating replay memory to store there transitions between states
        self.replayMemory = ReplayMemory.create()

        self.replayMemory:__init(replayMemoryArgs)

    else
        self.model = self:getNeuralNetwork():cl()
        local replayMemoryArgs = {maxSize = self.replayMemoryMaxSize,
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

    -- Rember self best network
    self.bestNetworkMaxFilename = (self.modelFilename or model) .. ("-best-max")
    self.bestNetworkAvgFilename = (self.modelFilename or model) .. ("-best-avg")
    self.bestNetworkAvgReward = 0
    self.bestNetworkMaxReward = 0


    -- Max rewards
	self.maxReward = 500
	self.minReward = 0---self.maxReward
    -- Bonus points and time penalty
	self.winBonus = self.maxReward
	self.timePenalty = 0
    -- Reward range (-a, a), used for scaling reward
    self.rewardRange = 1

	-- parameters, gradParameters of our model
	self.w, self.dw = self.model:getParameters()
    -- Auxiliary parameters
	self.deltas = self.dw:clone():fill(0)
	self.tmp = self.dw:clone():fill(0)
	self.g = self.dw:clone():fill(0)
	self.g2 = self.dw:clone():fill(0)

    -- Statistics
    self.validationSize = 10
    self.validationRate = 100
    self.validationEp = 0
    self.accuracyAvg = 0
    self.accuraccyMaxErr = 0
    self.accuraccyMinErr = 0
    self.accuracyMeanErr = 0
    self.predAvg = 0
    self.validationCounter = 0
end


-- This function returns the model of our neural network
function DeepQNN:getNeuralNetwork(hidden)
    local hus = hidden or 1024--144
    local net = nn.Sequential()
    -- 3 in, 48 out, kernel 8x8, step=4, output is floor((112 -8)/4 + 1) = 27
    net:add(nn.SpatialConvolutionMM(3, 48, 8, 8, 4, 4, 0, 0))
    net:add(nn.Tanh())
    -- 48x27x27 in, 96x12x12 out (27-4)/2 + 1 = 12
    net:add(nn.SpatialConvolutionMM(48, 96, 4, 4, 2, 2, 0, 0))
    net:add(nn.Tanh())
    net:add(nn.View(96*12*12))
    net:add(nn.Linear(96*12*12, hus))
    net:add(nn.Tanh())
    net:add(nn.Linear(hus, hus))
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
    elseif maxi == 7 then
        -- Do nothing take no action button pressed
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
    else
        return 7
    end
end


-- This function measure performance of game by net
function DeepQNN:measurePerformance(nTests)
    print("Measuring performance of network in a game.. This might take a while...")
    local maxRewardAchieved = 0
    local avgRewardAchieved = 0
    -- Performin nTests games
    for i = 1, nTests do
        collectgarbage()
        local endOfGame = false
        self.gameDriver:loadSaveState()
        xlua.progress(i, nTests)
        k = 1
        -- Playing until game over
        while (not endOfGame) do
            local state = self.gameDriver:getState(k)
            endOfGame = state.terminal
            if endOfGame then
                -- Remember max score, and avg score
                maxRewardAchieved = math.max(maxRewardAchieved, state.score)
                avgRewardAchieved = avgRewardAchieved + state.score
            else
                local action = self:eGreedyLearn(state)
                self.gameDriver:sendButtons(action)
                k = self.gameDriver:advanceToNextFrame(k)
            end
        end
    end

    avgRewardAchieved = avgRewardAchieved / nTests

    if maxRewardAchieved > self.bestNetworkMaxReward then
        torch.save(self.bestNetworkMaxFilename .. ".t7", {
            model = self.model,
            epochs = self.passedEpochs,
            minibatchSize = self.minibatchSize
        }, "binary", true)
        self.bestNetworkMaxReward = maxRewardAchieved
    end
    if avgRewardAchieved > self.bestNetworkAvgReward then
        torch.save(self.bestNetworkAvgFilename .. ".t7", {
            model = self.model,
            epochs = self.passedEpochs,
            minibatchSize = self.minibatchSize
        }, "binary", true)
        self.bestNetworkAvgReward = avgRewardAchieved
    end

    local f = io.open("performance-logs.txt", "a")
    f:write("Epoch : " .. self.passedEpochs .. 
        " , Step: " .. self.steps .. ", Avg:" ..
        avgRewardAchieved .. ", Max: " .. 
        maxRewardAchieved .. "\n")
    f:close()

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
            -- Getting set of keys to press
            local action = self:greedyLearn(state)
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


-- We are checking the error on whole memory.
function DeepQNN:testWholeReplayMemory()
    print("Testing... This takes a while...")
    local l = 1
    local k = 0
    local all = false
    local accMinErr = 10000000
    local accMaxErr = 0
    local accMeanErr = 0
    local accSumErr = 0
    while(not all) do
        collectgarbage()
        local a, r, s, s2, t = self.replayMemory:getMinibatch(l)
        xlua.progress(l, self.replayMemory.numEntries)
        if a == nil then
            all = true
        else
            self.minibatch = {a=a, r=r, s=s, s2=s2, t=t}
            local targets, delta, q2Max = self:updateQValues()
            accMinErr = math.min(accMinErr, delta:clone():abs():min())
            accMaxErr = math.max(accMaxErr, delta:clone():abs():max())
            accMeanErr = accMeanErr + delta:clone():abs():mean()
            accSumErr = accSumErr + delta:clone():abs():sum()
            k = k + 1
            l = l + self.minibatchSize
        end
    end
    xlua.progress(self.replayMemory.numEntries, self.replayMemory.numEntries)
    accMeanErr = accMeanErr / k
    local f = io.open("error-logs.txt", "a")
    f:write("Epoch : " .. self.passedEpochs .. 
        " , Step: " .. self.steps .. ", Mean:" ..
        accMeanErr .. ", Min: " .. 
        accMinErr .. ", Max: " ..
        accMaxErr .. ", Sum: " .. 
        accSumErr .. "\n")
    f:close()
end



-- Calculates the mean answer on the samples
function DeepQNN:updateStatistics()
    self:createRandomMiniBatch()
    local targets, delta, q2Max = self:updateQValues()
    self.accuraccyMinErr = delta:clone():abs():min()
    self.accuraccyMaxErr = delta:clone():abs():max()
    self.accuracyMeanErr = delta:clone():abs():mean()
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
            collectgarbage()
            self.steps = t
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
            if (self.validationCounter == self.validationRate) then
                self:updateStatistics()
                --print(string.format("%f", self.accuracyAvg))
                local f = io.open("logs.txt", "a")
                f:write("Epoch : " .. self.passedEpochs .. 
                    " , Step: " .. self.steps .. ", Mean:" ..
                    self.accuracyMeanErr .. ", Min: " .. 
                    self.accuraccyMinErr .. ", Max: " ..
                    self.accuraccyMaxErr .. "\n")
                f:close()
                --if (self.validationEp % 10 == 0) then
                --    f = io.open("log" .. self.validationEp .. ".txt", "w")
                --    f:write("Epoch : " .. self.passedEpochs .. 
                --    " , Step: " .. self.steps .. ", Error:" ..
                --    self.accuracyAvg .. "\n")
                --    f:close()
                --end
                --self.validationEp = self.validationEp + 1
                --print(string.format("%f", self.predAvg))
                self.validationCounter = 0
            end
            self.validationCounter = self.validationCounter + 1
			start_state = next_state
        end
        self.passedEpochs = self.passedEpochs + 1
        -- At the end of an epoch we save our network
        print("Passed epoch : " .. self.passedEpochs)
        self:saveNeuralNetwork()
        self:testWholeReplayMemory()
        collectgarbage()
        -- Measuring performance of the network
        self:measurePerformance(10)
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
    q2Max = self.model:forward(s2:cl()):max(2)
	-- compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
	q2 = q2Max:clone():mul(self.discount):cmul(t:cl())

	-- compute delta = r + (1-terminal) * gamma * max_a Q(s2, a)
	delta = r:clone():cl()
	delta:add(q2):cl()

	-- compute q = Q(s,a)
    local q_all = self.model:forward(s:cl())
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
    self.epsilon = math.max(self.epsilon_end, (self.epsilon_start / 
        (((self.passedEpochs * self.levelScale) + self.steps*self.stepW) + 1)))
    --self.epsilon = math.max(self.epsilon_end, self.epsilon_start - 
        --((self.passedEpochs * self.levelScale) + self.steps*self.stepW))
	if torch.uniform() < self.epsilon then
		return self:getAction(torch.rand(#self.classes))
	else
		return self:greedyLearn(state)
	end
end

-- Returns an action based on the greedy alg
function DeepQNN:greedyLearn(state)
    local output = self.model:forward(state.screenTensor:cl())
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
	--self.dw:add(-self.l2, self.w)

    local tim = math.max(0, (self.steps + self.passedEpochs * 100000))
    self.learning_rate = ((self.learning_rate_start - self.learning_rate_end) * 
        (self.learning_rate_endt - tim) / self.learning_rate_endt) + self.learning_rate_end
    self.learning_rate = math.max(self.learning_rate, self.learning_rate_end)

	-- Using gradients RMSprop
	self.g:mul(0.9):add(0.1, self.dw)
	self.tmp:cmul(self.dw, self.dw)
	self.g2:mul(0.9):add(0.1, self.tmp)
	self.tmp:cmul(self.g, self.g)
	self.tmp:mul(-1)
	self.tmp:add(self.g2)
    -- Smoothing value
	self.tmp:add(0.002)
	self.tmp:sqrt()

	-- Accumulating update
	self.deltas:mul(0):addcdiv(self.learning_rate, self.dw, self.tmp)
	self.w:add(self.deltas)
end

function DeepQNN:saveNeuralNetwork()
    local filename = ""
    if (self.lastSavedBackup == true) then
        filename = self.modelFilename
    else
        filename = self.modelBackupFilename
    end
    filename = filename .. "-network"
    torch.save(filename .. ".t7", {
        model = self.model,
        epochs = self.passedEpochs,
        minibatchSize = self.minibatchSize
    }, "binary", true)
    self.lastSavedBackup = not self.lastSavedBackup
end

function DeepQNN:saveReplayMemory()
    local filename = ""
    if (self.lastSavedBackup == true) then
        filename = self.modelFilename
    else
        filename = self.modelBackupFilename
    end
    filename = filename .. "-replayMemory"
    torch.save(filename .. ".t7", {
        replayMemory = self.replayMemory,
    }, "binary", true)
    --self.lastSavedBackup = not self.lastSavedBackup
end

function DeepQNN:loadNeuralNetwork()
    local saveModel = torch.load(self.modelFilenameLoad .. "-backup" 
        .. "-network" .. ".t7", "binary", true)
    self.model = saveModel.model--:float()
    --self.replayMemory = saveModel.replayMemory
    --setmetatable(self.replayMemory, ReplayMemory)
    --self.replayMemory.maxSize = self.replayMemoryMaxSize
    self.passedEpochs = saveModel.epochs
    self.minibatchSize = saveModel.minibatchSize
    --self.model = self.model:cl()
end

function DeepQNN:loadReplayMemory()
    local saveModel = torch.load(self.modelFilenameLoad .. "-replayMemory" 
        .. ".t7", "binary", true)
    self.replayMemory = saveModel.replayMemory
    setmetatable(self.replayMemory, ReplayMemory)
    self.replayMemory.maxSize = self.replayMemoryMaxSize
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
		--if not nextState.isLevelWon then
		--	return -1 * self.rewardRange
		--else
		--	reward = self.winBonus
		--end
        return 0
	end

	local delta_score = nextState.score - startState.score
	reward = delta_score + self.timePenalty
    --reward = nextState.score
	reward = math.min(self.maxReward, reward)
	reward = math.max(self.minReward, reward)
    return reward--(reward / self.maxReward) * self.rewardRange
end

net = DeepQNN.create()
args = {modelFilenameLoad="model20", modelFilename="model20", load=true}
net:__init(args)
net:train(1000,10000)
--net:test(10000)
