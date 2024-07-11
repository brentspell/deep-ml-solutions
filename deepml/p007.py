import numpy as np
import pytest


def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]]:
    if np.linalg.det(T) == 0:
        raise ValueError("T_singular")
    if np.linalg.det(S) == 0:
        raise ValueError("S_singular")

    return (np.linalg.inv(T) @ A @ S).tolist()


def test_transform_matrix() -> None:
    A: list[list[int | float]] = [[1, 2], [3, 4]]
    T: list[list[int | float]] = [[2, 0], [0, 2]]
    S: list[list[int | float]] = [[1, 0], [0, 1]]

    T_invalid: list[list[int | float]] = [[2, 2], [2, 2]]
    with pytest.raises(ValueError, match="T_singular"):
        transform_matrix(A, T_invalid, S)

    S_invalid: list[list[int | float]] = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="S_singular"):
        transform_matrix(A, T, S_invalid)

    assert transform_matrix(A, T, S) == [[0.5, 1.0], [1.5, 2.0]]
