import math


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # find the coefficients of the quadratic equation formed by the
    # characteristic equation |A - λI| = 0
    #    |[(a00 - λ) a01       | = 0
    #    | a10       (a11 - λ)]|
    # =>
    #    (a00 - λ) * (a11 - λ) - (a01 * a10) = 0
    #    λ^2 + -(a00 + a11) * λ + a00 * a11 - a01 * a10 = 0

    a = 1
    b = -(matrix[0][0] + matrix[1][1])
    c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # solve the equation using the quadratic formula
    d = math.sqrt(b**2 - 4 * a * c)
    e1 = (-b + d) / (2 * a)
    e2 = (-b - d) / (2 * a)

    # return the sorted eigenvalues
    return sorted([e1, e2], reverse=True)


def test_calculate_eigenvalues() -> None:
    matrix: list[list[float | int]] = [[2, 1], [1, 2]]
    assert calculate_eigenvalues(matrix) == [3.0, 1.0]
