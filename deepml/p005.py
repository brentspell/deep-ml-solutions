def scalar_multiply(
    matrix: list[list[int | float]],
    scalar: int | float,
) -> list[list[int | float]]:
    return [[c * scalar for c in r] for r in matrix]


def test_scalar_multiply() -> None:
    matrix: list[list[int | float]] = [[1, 2], [3, 4]]
    scalar = 2
    assert scalar_multiply(matrix, scalar) == [[2, 4], [6, 8]]
