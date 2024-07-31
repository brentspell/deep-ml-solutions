import math


def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float,
) -> tuple[list[float], float]:
    probabilities = [
        sigmoid(sum(w * f for w, f in zip(weights, f)) + bias) for f in features
    ]
    mse = sum(
        (y_true - y_pred) ** 2 for y_true, y_pred in zip(labels, probabilities)
    ) / len(labels)
    return [round(p, 4) for p in probabilities], round(mse, 4)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_single_neuron_model() -> None:
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0, 1, 0]
    weights = [0.7, -0.4]
    bias = -0.1
    probabilities, mse = single_neuron_model(features, labels, weights, bias)
    assert probabilities == [0.4626, 0.4134, 0.6682]
    assert mse == 0.3349
