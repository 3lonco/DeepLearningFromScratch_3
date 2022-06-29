# Add path for CI/CD tool
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../steps/")
)

import numpy as np
import pytest
import step06


def test_step06_backward():
    A = step06.Square()
    B = step06.Exp()
    C = step06.Square()

    # Start Forward
    x = step06.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    # end forward()

    # start Backward()
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    ans = 3.297442541400256
    assert x.grad == pytest.approx(ans)


def test_step06_single_backward():
    # Start Forward
    x = step06.Variable(np.array(1.0))
    A = step06.Exp()
    y = A(x)
    print(y)

    # start backward()
    y.grad = np.array(1.0)
    x.grad = A.backward(y.grad)
    print(x.grad)


def test_step06_function_forward():
    x = step06.Variable(np.array(1.0))
    y = step06.Function()
    # Raise NotImplementedError when you give str into step_function
    with pytest.raises(NotImplementedError) as e:
        f = y(x)


def test_step06_function_backward():
    x = step06.Variable(np.array(1.0))
    y = step06.Function()
    # Raise NotImplementedError when you give str into step_function
    with pytest.raises(NotImplementedError) as e:
        y.backward("test")
