import math


def activation(x: float) -> float:
    return 1 / (1 + math.e ** -x)


def activation_derivatives(x: float):
    return 1 - x


def errorFunc(target, actual: float) -> float:
    return (target - actual) ** 2


def errorFunc_derivatives(target, actual: float):
    return 2 * (target - actual)


def dot(inputs, weights: list[float]) -> float:
    s = 0
    for j in range(len(inputs)):
        s += inputs[j] * weights[j]
    return s


def create_inputs(layers: tuple[int]) -> list[list[float]]:
    inputs: list[list[float]] = []
    for i in range(len(layers)):
        inputs.append([])
        for j in range(layers[i]):
            inputs[i].append(0.0)
    return inputs


def create_weights(layers: tuple[int]) -> list[list[list[float]]]:
    w = []
    for i in range(len(layers) - 1):
        w.append([])
        for j in range(layers[i + 1]):
            w[i].append([0.5] * layers[i])
    return w


class Net:

    def __init__(self, *layers):
        self.layers: list[int] = list(layers)
        self.layers_count: int = len(self.layers)
        self.learning_rate: float = 0.1

        self.errors: list[float] = []
        self.totalError: float = 0

        self.inputs: list[list[float]] = create_inputs(layers)
        self.weights: list[list[list[float]]] = [
            [[0.13, 0.23],
             [0.14, 0.24],
             [0.15, 0.25]],

            [[0.36, 0.46, 0.56],
             [0.37, 0.47, 0.57]]
        ]

    def get_neuron_in_layer_count(self, n: int) -> int:
        return self.layers[n]

    def get_last_layer_len(self) -> int:
        return len(self.layers) - 1

    def get_last_layer_neuron_count(self) -> int:
        return self.layers[len(self.layers) - 1]

    def learn(self, train_x, train_y: list[list[float]]) -> None:
        self.errors = [0.0] * len(train_y)
        for i in range(len(train_x)):
            self.assign_inputs(train_x[i])
            self.forward()
            self.sample_error(i, train_y[i])
        self.totalError = sum(self.errors)
        self.back_propagation()

    def assign_inputs(self, xs: list[float]):
        self.inputs[0] = xs

    def forward(self) -> None:
        for i in range(self.layers_count - 1):
            for j in range(self.get_neuron_in_layer_count(i + 1)):
                self.inputs[i + 1][j] = dot(self.inputs[i], self.weights[i][j])
                self.inputs[i + 1][j] = activation(self.inputs[i + 1][j])

    def sample_error(self, index: int, train_y: list[float]) -> None:
        for i in range(len(train_y)):
            self.errors[index] += errorFunc(train_y[i], self.inputs[-1][i]) / len(train_y)

    def back_propagation(self) -> None:
        pass

    def local_errors(self):
        pass

    def update_weights(self):
        pass


trains: list[list[float]] = [
    [.1, .2],
    [.3, .4],
    [.5, .6],
    [.7, .8]
]

targets: list[list[float]] = [
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0]
]

net = Net(2, 3, 2)
net.learn(trains, targets)
