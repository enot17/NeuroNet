import math
import random as rnd


def activationFunction(x):
    if x <= 0:
        return 0
    return 1


class Neuron:

    def __init__(self, inputs):
        self.output: float = 0
        if type(inputs) == int:
            self.inputs = [inputs]
            self.weights = [rnd.random()]
        else:
            self.inputs = inputs
            self.weights = []
            for i in range(len(inputs)):
                self.weights.append(rnd.random())

    def activation(self):
        for x in range(len(self.inputs)):
            self.output += self.inputs[x] * self.weights[0]
        self.output = activationFunction(self.output)


class Layer:

    def __init__(self, n):
        self.size: int = n
        self.neurons: list[Neuron] = []

    def calkNeurons(self):
        for n in self.neurons:
            n.activation()

    def getInputs(self) -> list[float]:
        l = []
        for i in range(self.size):
            l.append(self.neurons[i].output)
        return l

    def __len__(self):
        return len(self.neurons)


class Net:

    def __init__(self, *lays):
        self.size = len(lays)
        self.layers = []
        for l in lays:
            self.layers.append(Layer(l))

    def fit(self, xs, targets):
        for i in range(len(xs)):
            self.assignInputs(xs[i])
            self.forward()
            er = self.error(targets[i])
            self.backPropagation(er)

    def assignInputs(self, x):
        for i in range(len(x)):
            self.layers[0].neurons.append(Neuron(x[i]))

    def forward(self):
        self.layers[0].calkNeurons()
        for i in range(1, self.size):
            prev = self.layers[i - 1].getInputs()
            for j in range(self.layers[i].size):
                self.layers[i].neurons.append(Neuron(prev))
            self.layers[i].calkNeurons()

    def error(self, target):
        actual = 0
        for neuron in self.layers[self.size - 1].neurons:
            if actual < neuron.output:
                actual = neuron.output
        return (target - actual) ** 2

    def backPropagation(self, error):
        pass

    def predict(self, x):
        return []


net = Net(2, 3, 1)
net.fit([
    [10, 11],
    [12, 21],
    [13, 31],
    [14, 41]
],
    [111, 222, 333, 444]
)

result = net.predict([
    [1, 1],
    [0, 0]
])
print(result)

