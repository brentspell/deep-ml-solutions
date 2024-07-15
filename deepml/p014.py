import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]],
    y: list[float],
) -> list[float]:
    X_ = np.array(X)
    y_ = np.array(y)
    return (np.linalg.inv(X_.T @ X_) @ (X_.T @ y_)).round(4).tolist()


def test_linear_regression_normal_equation() -> None:
    X: list[list[float]] = [[1, 1], [1, 2], [1, 3]]
    y: list[float] = [1, 2, 3]
    linear_regression_normal_equation(X, y) == [0.0, 1.0]
