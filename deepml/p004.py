def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    M, N = len(matrix), len(matrix[0])
    if mode == "row":
        return [sum(matrix[i]) / N for i in range(M)]
    else:
        return [sum(matrix[i][j] for i in range(M)) / M for j in range(N)]


def test_calculate_matrix_mean() -> None:
    matrix: list[list[float]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert calculate_matrix_mean(matrix, "column") == [4.0, 5.0, 6.0]
    assert calculate_matrix_mean(matrix, "row") == [2.0, 5.0, 8.0]
