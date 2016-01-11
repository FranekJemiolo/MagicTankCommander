ReplayMemory = {}

ReplayMemory.__index = ReplayMemory

function boolToNum(val)
	if val then
		return 1
	else
		return 0
	end
end

function ReplayMemory.create()
	local rpmem = {}
	setmetatable(rpmem, ReplayMemory)
	return rpmem
end

function ReplayMemory:__init(args)
	self.maxSize = args.maxSize
	self.numEntries = 0
	self.entries = {}
	self.insertIndex = 0
	self.inputDim = args.inputDim
	self.batchSize = args.batchSize


	self.buf_input = torch.Tensor(self.batchSize, 3, 56, 64):fill(0)
	self.buf_input2 = torch.Tensor(self.batchSize, 3, 56, 64):fill(0)
	self.buf_action = torch.Tensor(self.batchSize):fill(0)
	self.buf_reward = torch.Tensor(self.batchSize):fill(0)
	self.buf_term = torch.Tensor(self.batchSize):fill(0)
	
end

-- Add transition into the table
function ReplayMemory:add(elem)
	if self.numEntries < self.maxSize then
		self.numEntries = self.numEntries + 1
	end

	self.insertIndex = self.insertIndex + 1
	if self.insertIndex > self.maxSize then
		self.insertIndex = 1
	end

	self.entries[self.insertIndex] = elem
end

-- Clear memory
function ReplayMemory:reset()
	self.numEntries = 0
	self.insertIndex = 0
end

-- Returns mini batch
function ReplayMemory:sample()
	if self.insertIndex > 0 then
		local index
		local i
		local numSamples = 0

		while (numSamples < ((self.batchSize) - 1)) do
			i = torch.random(1, self.numEntries)
			numSamples = numSamples + 1
			self.buf_action[numSamples] = self.entries[i].action
			self.buf_reward[numSamples] = self.entries[i].reward
			self.buf_term[numSamples] = 
				boolToNum(self.entries[i].next_state.terminal)
			self.buf_input[numSamples] = 
				self.entries[i].start_state.screenTensor:clone()
			self.buf_input2[numSamples] = 
				self.entries[i].next_state.screenTensor:clone()
		end

		self.buf_action[self.batchSize] = self.entries[self.insertIndex].action
		self.buf_reward[self.batchSize] = self.entries[self.insertIndex].reward
		self.buf_term[self.batchSize] = 
			boolToNum(self.entries[self.insertIndex].next_state.terminal)
		self.buf_input[self.batchSize] = 
			self.entries[self.insertIndex].start_state.screenTensor:clone()
		self.buf_input2[self.batchSize] = 
			self.entries[self.insertIndex].next_state.screenTensor:clone()


		return self.buf_action, self.buf_reward, self.buf_input, 
			self.buf_input2, self.buf_term
	else
		return nil
	end
end

-- Returns last transition
function ReplayMemory:lastTransition()
	if (self.insertIndex >= 1) then
		return self.entries[self.insertIndex]
	else
		return nil
	end
end

-- Returns number of entries
function ReplayMemory:size()
	return self.numEntries
end
