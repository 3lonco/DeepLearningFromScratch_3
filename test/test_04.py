# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step04


def test_step04():
    A = step04.Square()
    B = step04.Exp()
    C = step04.Square()

    x = step04.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)
    ans = 1.648721270700128
    assert y.data == pytest.approx(ans)


def test_step04_diff():
    f = step04.Square()
    x = step04.Variable(np.array(2.0))
    dy = step04.numerical_diff(f, x)
    ans = 4.0  # f(x)^ = 2x. f(2)^ = 4.0
    assert dy == pytest.approx(ans)

