"""Простая полносвязная нейросеть, обучающаяся решению XOR без сторонних библиотек."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List


Vector = List[float]
Matrix = List[Vector]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    y = tanh(x)
    return 1 - y * y


def matmul(a: Matrix, b: Matrix) -> Matrix:
    result = zeros(len(a), len(b[0]))
    for i in range(len(a)):
        for j in range(len(b[0])):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(len(b)))
    return result


def add_bias(matrix: Matrix, bias: Vector) -> Matrix:
    return [[matrix[i][j] + bias[j] for j in range(len(bias))] for i in range(len(matrix))]


def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def apply_activation(matrix: Matrix) -> Matrix:
    return [[tanh(value) for value in row] for row in matrix]


def apply_activation_derivative(matrix: Matrix) -> Matrix:
    return [[tanh_derivative(value) for value in row] for row in matrix]


def elementwise_mul(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scalar_mul(matrix: Matrix, scalar: float) -> Matrix:
    return [[value * scalar for value in row] for row in matrix]


def mean_squared_error(predictions: Matrix, targets: Matrix) -> float:
    total = 0.0
    count = 0
    for pred_row, target_row in zip(predictions, targets):
        for pred, target in zip(pred_row, target_row):
            diff = pred - target
            total += diff * diff
            count += 1
    return total / count


def mean_squared_error_grad(predictions: Matrix, targets: Matrix) -> Matrix:
    grad = zeros(len(predictions), len(predictions[0]))
    count = len(predictions) * len(predictions[0])
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            grad[i][j] = 2 * (predictions[i][j] - targets[i][j]) / count
    return grad


def subtract_in_place(vec: Vector, delta: Vector) -> None:
    for i in range(len(vec)):
        vec[i] -= delta[i]


def subtract_matrix_in_place(matrix: Matrix, delta: Matrix) -> None:
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] -= delta[i][j]


@dataclass
class Layer:
    weights: Matrix
    bias: Vector

    def forward(self, inputs: Matrix) -> Matrix:
        weighted = matmul(inputs, self.weights)
        return apply_activation(add_bias(weighted, self.bias))


@dataclass
class Network:
    hidden: Layer
    output: Layer

    def forward(self, inputs: Matrix) -> Matrix:
        hidden_output = self.hidden.forward(inputs)
        weighted = matmul(hidden_output, self.output.weights)
        logits = add_bias(weighted, self.output.bias)
        return apply_activation(logits)

    def parameters(self) -> Iterable[Matrix | Vector]:
        return [self.hidden.weights, self.hidden.bias, self.output.weights, self.output.bias]


def train(network: Network, inputs: Matrix, targets: Matrix, *, epochs: int, lr: float) -> List[float]:
    losses: List[float] = []
    for _ in range(epochs):
        hidden_raw = add_bias(matmul(inputs, network.hidden.weights), network.hidden.bias)
        hidden_output = apply_activation(hidden_raw)

        output_raw = add_bias(matmul(hidden_output, network.output.weights), network.output.bias)
        predictions = apply_activation(output_raw)

        loss = mean_squared_error(predictions, targets)
        losses.append(loss)

        grad_loss = mean_squared_error_grad(predictions, targets)
        grad_output = elementwise_mul(grad_loss, apply_activation_derivative(output_raw))

        grad_output_weights = matmul(transpose(hidden_output), grad_output)
        grad_output_bias = [sum(col) for col in transpose(grad_output)]

        grad_hidden = elementwise_mul(matmul(grad_output, transpose(network.output.weights)), apply_activation_derivative(hidden_raw))
        grad_hidden_weights = matmul(transpose(inputs), grad_hidden)
        grad_hidden_bias = [sum(col) for col in transpose(grad_hidden)]

        subtract_matrix_in_place(network.output.weights, scalar_mul(grad_output_weights, lr))
        subtract_in_place(network.output.bias, [lr * value for value in grad_output_bias])
        subtract_matrix_in_place(network.hidden.weights, scalar_mul(grad_hidden_weights, lr))
        subtract_in_place(network.hidden.bias, [lr * value for value in grad_hidden_bias])

    return losses


def main() -> None:
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [[0.0], [1.0], [1.0], [0.0]]

    hidden_layer = Layer(
        weights=[
            [0.5, -0.5],
            [0.5, -0.5],
        ],
        bias=[0.0, 0.0],
    )
    output_layer = Layer(
        weights=[
            [0.5],
            [-0.5],
        ],
        bias=[0.0],
    )

    network = Network(hidden=hidden_layer, output=output_layer)
    losses = train(network, inputs, targets, epochs=5000, lr=0.1)

    print(f"Финальная ошибка: {losses[-1]:.6f}")
    print("Предсказания:")
    for inp, pred in zip(inputs, network.forward(inputs)):
        print(f"{inp} -> {pred[0]:.4f}")


if __name__ == "__main__":
    main()
