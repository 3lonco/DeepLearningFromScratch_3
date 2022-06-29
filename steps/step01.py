class Function:
    def __call__(self, input):
        x = input.data  # Get a data
        y = x ** 2  # Calculate
        output = Variable(y)  # return as Variable
        return output


class Variable:
    def __init__(self, data):
        self.data = data
