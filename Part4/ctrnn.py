import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1-x))

class CTRNN():

    def __init__(self, size, inputsize, outputsize):
        self.Size = size                        # number of neurons in the network
        self.InputSize = inputsize
        self.OutputSize = outputsize
        self.Voltage = np.zeros(size)           # neuron activation vector
        self.TimeConstants = np.ones(size)       # time-constant vector
        self.Biases = np.zeros(size)              # bias vector
        self.Weights = np.zeros((size,size))     # weight matrix
        self.SensorWeights = np.zeros((inputsize,size))          # neuron output vector
        self.MotorWeights = np.zeros((size,outputsize))            # neuron output vector
        self.Output = np.zeros(size)            # neuron output vector
        self.Input = np.zeros(size)             # neuron output vector

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size))
        self.SensorWeights = np.random.uniform(-10,10,size=(self.InputSize, self.Size))
        self.MotorWeights = np.random.uniform(-10,10,size=(self.Size, self.OutputSize))
        self.Biases = np.random.uniform(-10,10,size=(self.Size))
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.Size))
        self.invTimeConstants = 1.0/self.TimeConstants

    def setParameters(self,genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax):
        k = 0
        for i in range(self.Size):
            for j in range(self.Size):                          # XXX
                self.Weights[i][j] = genotype[k]*WeightRange    # XXX
                k += 1
        for i in range(self.InputSize):
            for j in range(self.Size):
                self.SensorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            for j in range(self.OutputSize):
                self.MotorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            self.Biases[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.Size):
            self.TimeConstants[i] = ((genotype[k] + 1)/2)*(TimeConstMax-TimeConstMin) + TimeConstMin
            k += 1
        self.invTimeConstants = 1.0/self.TimeConstants

    def initializeState(self,v):
        self.Voltage = v
        self.Output = sigmoid(self.Voltage+self.Biases)

    def initializeOutput(self,o):
        self.Output = o
        self.Voltage = inv_sigmoid(o) - self.Biases

    def step(self,dt,i):
        self.Input = np.dot(self.SensorWeights.T, i) ###
        netinput = self.Input + np.dot(self.Weights.T, self.Output)
        self.Voltage += dt * (self.invTimeConstants*(-self.Voltage+netinput))
        self.Output = sigmoid(self.Voltage+self.Biases)

    def out(self):
        return sigmoid(np.dot(self.MotorWeights.T, self.Output))

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, sensorweights=self.SensorWeights, motorweights=self.MotorWeights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.SensorWeights = params['sensorweights'] 
        self.MotorWeights = params['motorweights'] 
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants