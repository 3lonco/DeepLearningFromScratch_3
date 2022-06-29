# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step07


def test_step07_recall():
    A = step07.Square()
    B = step07.Exp()
    C = step07.Square()

    x = step07.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # backward
    y.grad = np.array(1.0)
    y.backward()
    ans = 3.297442541400256
    assert x.grad == pytest.approx(ans)


def test_step07_backward():
    A = step07.Square()
    x = step07.Variable(np.array(1.0))
    y = A(x)

    assert y.creator == A


def test_step07_assertationError():
    A = step07.Square()
    x = step07.Variable(np.array(1.0))
    dummy = step07.Variable(np.array(2.0))
    y = A(x)
    with pytest.raises(AssertionError) as e:
        assert y.creator.input == dummy
