def reshape_matrix(
    a: list[list[int | float]],
    new_shape: tuple[int, int],
) -> list[list[int | float]]:
    # or just np.array(a).reshape(new_shape).tolist()

    N = len(a[0])

    b: list[list[int | float]] = []
    k = 0
    for _ in range(new_shape[0]):
        r = []
        for _ in range(new_shape[1]):
            r.append(a[k // N][k % N])
            k += 1
        b.append(r)

    return b


def test_reshape_matrix() -> None:
    a: list[list[int | float]] = [[1, 2, 3, 4], [5, 6, 7, 8]]
    new_shape: tuple[int, int] = (4, 2)
    assert reshape_matrix(a, new_shape) == [[1, 2], [3, 4], [5, 6], [7, 8]]
