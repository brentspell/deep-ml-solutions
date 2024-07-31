import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []
    for _ in range(epochs):
        y = sigmoid(features @ updated_weights + updated_bias)
        loss = ((labels - y) ** 2).mean()

        g_l = y - labels
        g_s = y * (1.0 - y) * g_l
        g_b = g_s.sum() / g_s.shape[0]
        g_w = (g_s @ features) / g_s.shape[0]

        updated_weights -= learning_rate * g_w
        updated_bias -= learning_rate * g_b

        mse_values.append(round(loss, 4))

    return updated_weights.round(4), round(updated_bias, 4), mse_values


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def test_train_neuron() -> None:
    features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = np.array([1, 0, 0])
    initial_weights = np.array([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2
    output_weights, output_bias, output_mse_values = train_neuron(
        features,
        labels,
        initial_weights,
        initial_bias,
        learning_rate,
        epochs,
    )
    expect_weights = np.array([0.1019, -0.1711])
    expect_bias = -0.0083
    expect_mse_values = [0.3033, 0.2987]
    assert np.allclose(output_weights, expect_weights)
    assert np.allclose(output_bias, expect_bias)
    assert np.allclose(output_mse_values, expect_mse_values)
