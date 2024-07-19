import numpy as np


def cross_validation_split(data: np.ndarray, k: int) -> list:
    m = data.shape[0]
    f = m // k
    s = np.arange(k) * f
    return [
        [np.concatenate([data[:lh], data[rh:]]).tolist(), data[lh:rh].tolist()]
        for lh, rh in zip(s, np.append(s[1:], m))
    ]


def test_cross_validation_split() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 5
    output = cross_validation_split(data, k)
    expect = [
        [[[3, 4], [5, 6], [7, 8], [9, 10]], [[1, 2]]],
        [[[1, 2], [5, 6], [7, 8], [9, 10]], [[3, 4]]],
        [[[1, 2], [3, 4], [7, 8], [9, 10]], [[5, 6]]],
        [[[1, 2], [3, 4], [5, 6], [9, 10]], [[7, 8]]],
        [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10]]],
    ]
    assert output == expect
