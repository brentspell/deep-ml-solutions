def determinant_4x4(matrix: list[list[int | float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        return sum(
            (-1) ** k
            * matrix[k][0]
            * determinant_4x4(
                [
                    [matrix[i][j] for j in range(1, len(matrix[0]))]
                    for i in range(len(matrix))
                    if i != k
                ]
            )
            for k in range(len(matrix))
        )


def test_determinant_4x4() -> None:
    A: list[list[int | float]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    assert determinant_4x4(A) == 0
