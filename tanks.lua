-- This program was written by Franciszek Jemioło

-- We have to hold a variable for population, a list of gene invetions so that
-- the invention number does not grow too big.
-- In the population we have to create an ordered list of species / by fitness
-- and by structure (when creating new population).
-- We start with uniform population of one species that has no hidden nodes
-- mutateConnections - add one new connection - this is an invetion so has to be
-- kept there and we have to look if it isn't there when creating new.
-- mutateNodes - actually it just adds two connection as an invetion so we 
-- also have to check there to see if it isn't already there, and only then
-- assign new number to it.
-- We probably have to go through the invetions and after that choose a random
-- values that are not already in inventions.
-- This piece of magic requires torch for the tensor library so it will be
-- easy to handle tensor and matrix operations
-- Mutations only occur between input and output layer! (output is not mutated)
require('torch')


-- This function creates uniform population
function createUniformPopulation()
end

-- Setting manual seed for repeated results
torch.manualSeed(1)

inventionCounter = 1

transferFunction = 'ReLU'

-- This function create a new neuron that recieves input from inputCount neurons
-- and passes it to outputCount neurons
function createEmptyNeuron(inputCount, inputIDs, id)--outputCount, outputIDs, id)
    -- Create weights, initialized with random value    
    local inputWeights = {}
    for i = 1, inputCount do
        -- Weight value
        local w = torch.uniform()
        -- Id of a node from which the weight is coming
        local from = inputIDs[i]
        -- Is the connection disabled?
        local disabled = false
        inputWeights[i] = {value = w, from = from, disabled = disabled}
    end
    --local outputWeights = {}
    --for i = 1, outputCount do
    --    local w = torch.random()
    --    local to = outputIDs[i]
    --    local disabled = false
    --    outputWeights[i] = {value = w, to = to, disabled = disabled}
    --end
    -- Setting up node id so we know know what are connections (and what is the
    -- structure of a neural network)
    return {inputWeights = inputWeights, id = id}--outputWeights = outputWeights, id = id}
end

-- Just copying our neuron.
function copyNeuron(neuron)
    local inputWeights = {}
    local outputWeights = {}
    local id = neuron.id
    for i = 1, #neuron.inputWeights do
        local w = neuron.inputWeights[i].value
        local from = neuron.inputWeights[i].from
        local disabled = neuron.inputWeights[i].disabled
        inputWeights[i] = {value = w, from = from, disabled = disabled}
    end
    --for i = 1, #neuron.outputWeights do
    --    local w = neuron.outputWeights[i].value
    --    local to = neuron.outputWeights[i].to
    --    local disabled = neuron.outputWeights[i].disabled
    --    inputWeights[i] = {value = w, to = to, disabled = disabled}
    --end    
    return {inputWeights = inputWeights, id = id}--outputWeights = outputWeights, id = id}
end

function getTransferFunction()
    if transferFunction == 'ReLU' then
        return function (x)
            if x > 0 then
                return x
            else
                return 0
            end
        end
    end
end

function calculateOutputNeuron(neuron, input)
    local w = 0
    for i = 1, #input do
        if (not neuron.inputWeights[i].disabled) then
            w = w + input[i] * neuron.inputWeights[i].value
        end
    end
    return getTransferFunction()(w)
end
        
    
-- Function creates an empty neural network of wanted size, this is done only
-- on the start, new networks are created as offspring, no network will be 
-- created this way later, this is why the inventionCounter will be at good 
-- number even when creating multpile NN, also they will have uniform structure 
-- at start.
function createEmptyNN(inputCount, outputCount)
    local inputLayer = {}
    local outputLayer = {}
    local inputIDs = {}
    for i = 1, inputCount do
        inputIDs[i] = i
    end
    --local outputIDs = {}
    --for i = 1, outputCount do
        -- Because output nodes will be after input nodes in our table
    --    outputIDs[i] = inputCount + i
    --end
    -- Creating input layer
    for i = 1, inputCount do
        -- Takes only one input it percieves, and output to all output nodes
        local inIDs = {}
        inIDs[1] = i
        inputLayer[i] = createEmptyNeuron(1, inIDs, i) --outputCount, outputIDs, i)
    end
    -- Creating output layer
    for i = 1, outputCount do
        -- Connected to all inputs, and to only of it's output
        --local outIDs = {}
        --outIDs[1] = i + inputCount
        outputLayer[i] = createEmptyNeuron(inputCount, inputIDs, i)--1, outIDs, 
        --    i + inputCount)
    end
    -- Creating the genome for easier use of measurements and other control
    -- It keeps all of the connections of the neural network, it's pretty
    -- redundant but it probably will help to speed up the process
    local genome = {}
    inventionCounter = 1
    -- The input Layer connections // The output connections are not included
    for i = 1, #outputLayer do
        for j = 1, #outputLayer[i].inputWeights do
            genome[inventionCounter] = {from = outputLayer[i].inputWeights[j].from,
                to = outputLayer[i].id, 
                disabled = outputLayer[i].inputWeights[j].disabled}
            inventionCounter = inventionCounter + 1
        end
    end
    -- When mutation the genome will probably be changed so as the output 
    -- weights, and we will have to keep some track of it
    return {inputLayer = inputLayer, hiddenLayer = nil, 
        outputLayer = outputLayer, genome = genome}
end

-- This function creates a neural network based on a genomes of parents
function createOffspringNN(fatherGenome, fatherFitness, motherGenome, 
    motherFitness)
end

-- This function creates new gene invetion and assign to it a number
function createNewInvention()
end



Queue = {}
function Queue.new ()
    return {first = 0, last = -1}
end

function Queue.pushleft (queue, value)
    local first = queue.first - 1
    queue.first = first
    queue[first] = value
end

function Queue.pushright (queue, value)
    local queue = queue.last + 1
    queue.last = last
    queue[last] = value
end

function Queue.popleft (queue)
    local first = queue.first
    if first > queue.last then error("list is empty") end
    local value = queue[first]
    queue[first] = nil        -- to allow garbage collection
    queue.first = first + 1
    return value
end

function Queue.popright (queue)
    local last = queue.last
    if queue.first > last then error("list is empty") end
    local value = queue[last]
    queue[last] = nil         -- to allow garbage collection
    queue.last = last - 1
    return value
end

function Queue.isEmpty(queue)
    if (first > last) then
        return true
    else
        return false
    end
end


-- This function calculates output of a network
function calculateOutputNN(neuralNetwork, input)
    local inputOutput = {}
    for i = 1, #input do
        local x = {}
        x[1] = input[i]
        -- Assuming that input size equals neuralNetwork inputLayer size
        inputOutput[i] = 
            calculateOutputNeuron(neuralNetwork.inputLayer[i], x)
    end
    if neuralNetwork.hiddenLayer ~= nil then
        local q = Queue.new()
        for i = 1, #neuralNetwork.hiddenLayer do
            Queue.pushright(q, neuralNetwork[i])
        end
        while not Queue.isEmpty(q) do
            local n = Queue.popleft(q)
            local countable = true
            for i = 1, #n.inputWeights do
                if inputOutput[n.inputWeights[i].from] == nil then
                    countable = false
                    break
                end
            end
            -- If not countable, we put it at the end
            if not countable then
                Queue.pushright(q, n)
            else
            -- It is countable, we count the activation and insert it
                local x = {}
                for i = 1, #n.inputWeights do
                    x[i] = inputOutput[n.inputWeights[i].from]
                end
                inputOutput[n.id] = calculateOutputNeuron(n, x)
            end
        end
    end
    local output = {}
    for i = 1, #neuralNetwork.outputLayer do
        local x = {}
        for j = 1, #neuralNetwork.outputLayer[i].inputWeights do
            x[j] = inputOutput[
                neuralNetwork.outputLayer[i].inputWeights[j].from]
        end
        output[i] = calculateOutputNeuron(neuralNetwork.outputLayer[i], x)
    end
    return output
end

-- This function calculates fitness of a network
function calculateFitness(neuralNetwork)
end

-- This function calculates how two neural networks are compatible 
-- (in genome structure)
function measureNetworks(neuralNet1, neuralNet2)
end

-- This function mutates connections of a given neural network
function mutateConnections(neuralNetwork)
end

-- This function mutates nodes of a given neural network (adds a node)
function mutateNodes(neuralNetwork)
end

-- This function creates a new pool of population, it calculates new species
-- organism in the same species must share the fitness of their niche
function createNewPopulation(previousPopulation)
end

-- This function is purely for testing of modules
function testNN()
    neuralNet = createEmptyNN(64, 4)
    input = {}
    for i = 1, 64 do
        input[i] = 1
    end
    output = calculateOutputNN(neuralNet, input)
    for i = 1, 4 do
        print(output[i])
    end
end

testNN()


