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

inventions = {}

transferFunction = 'ReLU'

-- This function create a new neuron that recieves input from inputCount neurons
function createEmptyNeuron(inputCount, inputIDs, id)
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
    -- Setting up node id so we know know what are connections (and what is the
    -- structure of a neural network)
    return {inputWeights = inputWeights, id = id}
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
    return {inputWeights = inputWeights, id = id}

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

-- This function creates initial inventions // connections, from initial net
function createInitialInventions(neuralNetwork)
    for i = 1, #neuralNetwork.genome do
        inventions[i] = {from = neuralNetwork.genome[i].from, 
            to = neuralNetwork.genome[i].to}
    end
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
    -- Creating input layer
    for i = 1, inputCount do
        -- Takes only one input it percieves, and output to all output nodes
        local inIDs = {}
        inIDs[1] = i
        inputLayer[i] = createEmptyNeuron(1, inIDs, i)
    end
    -- Creating output layer
    for i = 1, outputCount do
        -- Connected to all inputs, and to only of it's output
        outputLayer[i] = createEmptyNeuron(inputCount, inputIDs, i)
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
-- Before calling this function we should check if the parents match.
function createOffspringNN(fatherGenome, fatherFitness, motherGenome, 
    motherFitness)
    
    
end

-- This function creates new gene invention and assign to it a number
function createNewInvention()
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
            -- It is countable, we count the output and insert it
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
    neuralNet = createEmptyNN(4096, 64)
    input = {}
    for i = 1, 4096 do
        input[i] = torch.uniform()
    end
    output = calculateOutputNN(neuralNet, input)
    for i = 1, 64 do
        print(output[i])
    end
end

testNN()


