import numpy as np


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Function:
    def __call__(self, input):
        x = input.data  # Get a data
        y = self.forward(x)  # Calculate
        output = Variable(y)  # return as Variable
        output.set_creator(self)

        self.input = input  # Remember the input value
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad =None
        self.creator = None
    
    def set_creator(self,func):
        self.creator = func

    def backward(self):
        funcs =[self.creator]
        while funcs:
            f= funcs.pop() # get function
            x,y=f.input,f.output # get input and output of function
            x.grad = f.backward(y.grad) #call backward() method

            if x.creator is not None:
                funcs.append(x.creator) #Add a function one before itself to the list
