# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step02


def test_step02():
    x = step02.Variable(np.array(1.0))
    y = step02.Function()
    # Raise NotImplementedError when you give str into step_function
    with pytest.raises(NotImplementedError) as e:
        f = y(x)


def test_step02_squre():
    x = step02.Variable(np.array(1.0))
    # Raise TypeError when you give str into step_function
    f = step02.Square()
    y = f(x)
    print(type(y))
