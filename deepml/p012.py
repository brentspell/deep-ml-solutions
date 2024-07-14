import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    # https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd/19646#19646

    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    t1 = x1 / h1
    t2 = x2 / h2

    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    if h1 != h2:
        V = np.diag(1.0 / s) @ U.T @ A
    else:
        V = np.diag([1.0 / s[0], 0]) @ U.T @ A

    return U, s, V


def test_svd_2x2_singular_values() -> None:
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    U_expect, s_expect, V_expect = np.linalg.svd(A)
    U_actual, s_actual, V_actual = svd_2x2_singular_values(A)
    assert np.allclose(U_expect, U_actual)
    assert np.allclose(s_expect, s_actual)
    assert np.allclose(V_expect, V_actual)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    U_expect, s_expect, V_expect = np.linalg.svd(A)
    U_actual, s_actual, V_actual = svd_2x2_singular_values(A)
    assert np.allclose(U_expect, U_actual)
    assert np.allclose(s_expect, s_actual)
    assert np.allclose(V_expect, V_actual)
