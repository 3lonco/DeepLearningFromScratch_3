class Function:
    def __init__(self, input):
        x = input.data  # Get a data
        y = self.forward(x)  # Calculate
        output = Variable(y)  # return as Variable
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


class Variable:
    def __init__(self, data):
        self.data = data
