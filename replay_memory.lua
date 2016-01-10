ReplayMemory = {}

function ReplayMemory:__init(args)
	self.maxSize = args.maxSize
	self.numEntries = 0
	self.insertIndex = 0
	self.inputDim = args.inputDim
	self.batchSize = args.batchSize

	self.buf_input = torch.ByteTensor(self.batchSize):fill(0)
	self.buf_input2 = torch.ByteTensor(self.batchSize):fill(0)
	self.buf_action = torch.LongTensor(self.batchSize):fill(0)
	self.buf_reward = torch.zeros(self.batchSize)
	self.buf_term = torch.ByteTensor(self.batchSize)
end

-- Add transition into the table
function ReplayMemory.add(elem)
	if self.numEntries < self.maxSize then
		self.numEntries = self.numEntries + 1
	end

	self.insertIndex = self.insertIndex + 1
	if self.insertIndex > self.maxSize then
		self.insertIndex = 1
	end

	ReplayMemory[insertIndex] = elem
end

-- Clear memory
function ReplayMemory:reset()
	self.numEntries = 0
	self.insertIndex = 0
end

-- Returns mini batch
function ReplayMemory:sample()
	local index
	local i
	local numSamples = 0

	while numSamples < self.batchSize-1 do
		i = torch.random(1, self.numEntries)
		numSamples = numSamples + 1

		buf_action[numSamples] = ReplayMemory[i].action
		buf_reward[numSamples] = ReplayMemory[i].reward
		buf_term[numSamples] = ReplayMemory[i].next_state.terminal
		buf_input[numSamples] = ReplayMemory[i].start_state.screenTensor
		buf_input2[numSamples] = ReplayMemory[i].next_state.screenTensor
	end

	-- add last transition to the mini batch
	local last = self.lastTransition()
	buf_action[batchSize] = last.action
	buf_reward[batchSize] = last.reward
	buf_term[batchSize] = last.next_state.terminal
	buf_input[batchSize] = last.start_state.screenTensor
	buf_input2[batchSize] = last.next_state.screenTensor

	return buf_action, buf_reward, buf_input, buf_input2, buf_term
end

-- Returns last transition
function ReplayMemory:lastTransition()
	return ReplayMemory[self.insertIndex]
end

-- Returns number of entries
function ReplayMemory:size()
	return self.numEntries
end
