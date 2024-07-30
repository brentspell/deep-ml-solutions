import numpy as np


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel: str = "linear",
    lambda_val: float = 0.01,
    iterations: int = 100,
    sigma: float = 1.0,
) -> tuple[list, float]:
    alphas = np.zeros([data.shape[0]])
    b = 0.0
    k = k_rbf if kernel == "rbf" else k_linear
    for t in range(iterations):
        for i in range(data.shape[0]):
            d = (
                sum(
                    alphas[j] * labels[j] * k(data[i], data[j], sigma=sigma)
                    for j in range(data.shape[0])
                )
                + b
            )
            if labels[i] * d < 1.0:
                lr = 1 / (lambda_val * (t + 1))
                alphas[i] += lr * (labels[i] - lambda_val * alphas[i])
                b += lr * labels[i]

    return alphas.round(4).tolist(), np.round(b, 4)


def k_linear(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    return x @ y


def k_rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * sigma**2))


def test_pegasos_kernel_svm() -> None:
    data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]])
    labels = np.array([1, 1, -1, -1])
    kernel = "rbf"
    lambda_val = 0.01
    iterations = 100
    alphas, b = pegasos_kernel_svm(data, labels, kernel, lambda_val, iterations)
    assert alphas == [100.0, 99.0, -100.0, -100.0]
    assert b == -150.0
