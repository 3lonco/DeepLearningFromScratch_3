# Add path for CI/CD tool
import sys
import os
from typing import Type

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step09

ans = 3.297442541400256

def test_step09_TypeError():
    x = step09.Variable(np.array(1.0))
    x = step09.Variable(None)
    with pytest.raises(TypeError) as e:
        x=step09.Variable(1.0)


def test_step09_scholar():

    x = step09.Variable(np.array(0.5))
    y = step09.square(step09.exp(step09.square(x)))
    y.backward()
    assert x.grad == pytest.approx(ans)


def test_step09_scholar():

    x = step09.Variable(np.array(0.5))
    y = step09.square(step09.exp(step09.square(x)))
    y.backward()
    assert x.grad == pytest.approx(ans)


def test_step09_function():

    x = step09.Variable(np.array(0.5))
    y = step09.square(step09.exp(step09.square(x)))

    # backward
    y.grad = np.array(1.0)
    y.backward()
    assert x.grad == pytest.approx(ans)



def test_step09_functions():

    x = step09.Variable(np.array(0.5))
    a = step09.square(x)
    b = step09.exp(a)
    y = step09.square(b)

    # backward
    y.grad = np.array(1.0)
    y.backward()
    assert x.grad == pytest.approx(ans)
