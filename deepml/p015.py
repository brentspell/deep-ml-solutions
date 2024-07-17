import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int,
) -> np.ndarray:
    weights = np.zeros([X.shape[1]])

    for _ in range(iterations):
        loss = X @ weights - y
        grad = X.T @ loss / X.shape[0]
        weights -= alpha * grad

    return weights.round(4)


def test_linear_regression_gradient_descent() -> None:
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.01
    iterations = 1000
    output = linear_regression_gradient_descent(X, y, alpha, iterations)
    assert np.allclose(output, np.array([0.1107, 0.9513]))
