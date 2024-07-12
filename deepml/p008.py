def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    [a, b], [c, d] = matrix

    det = a * d - b * c
    if det == 0.0:
        return None

    return [
        [d / det, -b / det],
        [-c / det, a / det],
    ]


def test_inverse_2x2() -> None:
    # non-invertible matrix
    matrix = [[1.0, 1.0], [1.0, 1.0]]
    assert inverse_2x2(matrix) is None

    # invertible matrix
    matrix = [[4.0, 7.0], [2.0, 6.0]]
    assert inverse_2x2(matrix) == [[0.6, -0.7], [-0.2, 0.4]]
