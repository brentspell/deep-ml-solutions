import math


def sigmoid(z: float) -> float:
    return round(1.0 / (1.0 + math.exp(-z)), 4)


def test_sigmoid() -> None:
    assert sigmoid(0) == 0.5
