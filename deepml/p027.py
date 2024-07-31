import numpy as np


def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    p = np.array(B) @ np.linalg.inv(C)
    return p.round(4).tolist()


def test_transform_basis() -> None:
    B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    C = [[1.0, 2.3, 3.0], [4.4, 25.0, 6.0], [7.4, 8.0, 9.0]]
    output = transform_basis(B, C)
    expect = [
        [-0.6772, -0.0126, 0.2342],
        [-0.0184, 0.0505, -0.0275],
        [0.5732, -0.0345, -0.0569],
    ]
    assert np.allclose(output, expect)
