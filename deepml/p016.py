import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = data.mean(0, keepdims=True)
    si = data.std(0, keepdims=True)
    standardized_data = (data - mu) / si

    lo = data.min(0, keepdims=True)
    hi = data.max(0, keepdims=True)
    normalized_data = (data - lo) / (hi - lo)

    return standardized_data.round(4), normalized_data.round(4)


def test_feature_scaling() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]])
    s, n = feature_scaling(data)
    assert np.allclose(s, [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]])
    assert np.allclose(n, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
