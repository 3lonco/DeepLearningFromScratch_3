# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step08


def test_step08_recall():
    A = step08.Square()
    B = step08.Exp()
    C = step08.Square()

    x = step08.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # backward
    y.grad = np.array(1.0)
    y.backward()
    ans = 3.297442541400256
    assert x.grad == pytest.approx(ans)
