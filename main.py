import math
import random


def activationFunction(x) -> float:
    return 1 / (1 + math.e ** -x)


class Net:

    def __init__(self, *lays) -> None:
        self.inputs: list[list[float]] = []
        self.weights: list[list[list[float]]] = []
        self.errors: list[float] = []

        for i in lays:
            self.inputs.append([0.0] * i)

        for i in range(len(lays) - 1):
            self.weights.append([])
            for j in range(lays[i]):
                a = [j] * lays[i + 1]
                self.weights[i].append(a)

    def fit(self, train_inputs: list[list[float]],
            targets: list[list[float]],
            learning_rate: float
            ) -> None:
        for i in range(len(train_inputs)):
            self.inputs[0] = train_inputs[i]
            self.forward()
            self.calkError(targets[i])
            self.backPropagation(learning_rate)

    def forward(self):
        for a in range(1, len(self.inputs)):
            for c in range(len(self.inputs[a])):
                for b in range(len(self.inputs[a - 1])):
                    self.inputs[a][c] += self.inputs[a - 1][b] * self.weights[a - 1][b][c]
                self.inputs[a][c] = activationFunction(self.inputs[a][c])

    def calkError(self, targets: list[float]):
        for i in range(len(self.last())):
            self.errors.append(((self.last()[i] - targets[i]) ** 2) / len(self.last()))

    def backPropagation(self, lr: float):
        pass

    def predict(self,test:list[list[float]]):
        return []

    def last(self):
        return self.inputs[len(self.inputs) - 1]


net: Net = Net(2, 3, 2)
net.fit([
    [10, 20],
    [11, 22],
    [13, 23],
    [14, 24]
],
    [[1, 0],
     [1, 0],
     [1, 0],
     [0, 1]], 0.1
)

result = net.predict([
    [1, 1],
    [0, 0]
])
print(result)

