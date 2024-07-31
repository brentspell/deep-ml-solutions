import typing as T

import numpy as np


class Value:
    data: float | np.ndarray
    grad: float | np.ndarray

    def __init__(
        self,
        data: float | np.ndarray,
        children: tuple = (),
        op: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: T.Union["Value", float, np.ndarray]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        result = Value(data=self.data + other.data, children=(self, other), op="+")

        def backward() -> None:
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __mul__(self, other: T.Union["Value", float, np.ndarray]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        result = Value(data=self.data * other.data, children=(self, other), op="*")

        def backward() -> None:
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def relu(self) -> "Value":
        data = self.data if self.data > 0.0 else 0.0
        result = Value(data=data, children=(self,), op="ReLU")

        def backward() -> None:
            self.grad += result.grad if result.data > 0.0 else 0.0

        result._backward = backward
        return result

    def backward(self) -> None:
        done = set()

        def walk(node: "Value") -> T.Iterator[Value]:
            if node not in done:
                done.add(node)
                yield node

                for child in node._prev:
                    yield from walk(child)

        self.grad = 1.0
        for node in walk(self):
            node._backward()


def test_value() -> None:
    a = Value(2.0)
    b = Value(3.0)
    c = Value(10.0)
    d = a + b * c
    e = Value(7.0) * Value(2.0)
    f = e + d
    g = f.relu()
    g.backward()
    assert str(a) == "Value(data=2.0, grad=1.0)"
    assert str(b) == "Value(data=3.0, grad=10.0)"
    assert str(c) == "Value(data=10.0, grad=3.0)"
    assert str(d) == "Value(data=32.0, grad=1.0)"
    assert str(e) == "Value(data=14.0, grad=1.0)"
    assert str(f) == "Value(data=46.0, grad=1.0)"
    assert str(g) == "Value(data=46.0, grad=1.0)"
