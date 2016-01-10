ReplayMemory = {}

function ReplayMemory:__init(args)
	self.maxSize = args.maxSize
	self.numEntries = 0
	self.insertIndex = 0
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

-- Returns mini batch of the given size
function ReplayMemory:sample(batchSize)
	local eps = batchSize / self.numEntries
	local numSamples = 0
	local i = 1
	samples = {}
	while numSamples < batchSize do
		if torch.uniform() < 0.3 then
			numSamples = numSamples + 1
			samples[numSamples] = ReplayMemory[i]
		end
		i = i + 1
		if i > self.numEntries then
			i = 1
		end
	end
	return samples
end

-- Returns last transition
function ReplayMemory:lastTransition()
	return ReplayMemory[self.insertIndex]
end

-- Returns number of entries
function ReplayMemory:size()
	return self.numEntries
end
