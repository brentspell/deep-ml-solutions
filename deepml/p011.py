import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d = 1.0 / np.diag(A)
    LU = np.where(np.eye(A.shape[0]), 0, A)

    x = np.zeros([A.shape[0]])
    for i in range(n):
        x = d * (b - (LU * x).sum(1))

    return x.round(4).tolist()


def test_solve_jacobi() -> None:
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3])
    n = 2
    assert solve_jacobi(A, b, n) == [0.146, 0.2032, -0.5175]
