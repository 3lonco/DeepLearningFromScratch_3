# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step01


def test_step01():
    data = np.array(1.0)
    x = step01.Variable(data)
    assert x.data == 1.0


def test_step01_function():
    x = step01.Variable(np.array(10))
    f = step01.Function()
    y = f(x)
    print(type(y))
    print(y.data)
    assert y.data == 100 and type(y) is step01.Variable
