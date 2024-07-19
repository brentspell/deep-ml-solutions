import numpy as np


def pca(data: np.ndarray, k: int) -> list[list[int | float]]:
    x = data.copy().astype(np.float64)

    # standardize the dataset
    x -= x.mean(0, keepdims=True)
    x /= x.std(0, keepdims=True).clip(min=1e-5)

    # compute the covariance matrix and eigendecomposition
    cov = x.T @ x
    eva, eve = np.linalg.eig(cov)

    # take the top-k eigenvector features by the eigenvalues
    i = np.argsort(eva)[::-1][:k]
    return eve[:, i].round(4).tolist()


def test_pca() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 1
    assert pca(data, k) == [[0.7071], [0.7071]]
