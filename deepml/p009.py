def matrixmul(
    a: list[list[int | float]],
    b: list[list[int | float]],
) -> list[list[int | float]] | int:
    M = len(a)
    N = len(a[0]) if a else 0
    K = len(b[0]) if b else 0

    if N != len(b):
        return -1

    c: list[list[int | float]] = [[0] * K for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                c[i][k] += a[i][j] * b[j][k]

    return c


def test_matrixmul() -> None:
    # invalid matmul
    a: list[list[int | float]] = [[1], [3]]
    b: list[list[int | float]] = [[2, 1], [3, 4]]
    assert matrixmul(a, b) == -1

    # valid matmul
    a = [[1, 2], [2, 4]]
    b = [[2, 1], [3, 4]]
    assert matrixmul(a, b) == [[8, 9], [16, 18]]
