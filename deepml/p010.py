def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    N = len(vectors[0])
    m = [sum(v) / N for v in vectors]
    d = [[vi - mk for vi in v] for v, mk in zip(vectors, m)]
    c = [[sum(dik * djk for dik, djk in zip(di, dj)) / (N - 1) for dj in d] for di in d]
    return c


def test_calculate_covariance_matrix() -> None:
    vectors: list[list[float]] = [[1, 2, 3], [4, 5, 6]]
    assert calculate_covariance_matrix(vectors) == [[1.0, 1.0], [1.0, 1.0]]
