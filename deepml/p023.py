import math


def softmax(scores: list[float]) -> list[float]:
    e = [math.exp(x) for x in scores]
    s = sum(e)
    return [round(x / s, 4) for x in e]


def test_softmax() -> None:
    scores = [1.0, 2.0, 3.0]
    assert softmax(scores) == [0.0900, 0.2447, 0.6652]
