from random import uniform
from math import e
from copy import deepcopy

class Matrix(list):
    def __repr__(self):
        return 'Matrix({})'.format(super().__repr__())
    def __neg__(self):
        for index1, value1 in enumerate(self):
            for index2, value2 in enumerate(value1):
                self[index1][index2] = -value2
        return self
    def template(self, other, act, reverse=False):
        new_matrix = deepcopy(self)
        if isinstance(other, (int, float)):
            for index1, value1 in enumerate(self):
                for index2, value2 in enumerate(value1):
                    data = (value2, other) if not reverse else (other, value2)
                    new_matrix[index1][index2] = act(*data)
        else:
            if (len(self), len(self[0])) == (len(other), len(other[0])):
                for index1, value1 in enumerate(self):
                    for index2, value2 in enumerate(value1):
                        data = (value2, other[index1][index2]) if not reverse else (other[index1][index2], value2)
                        new_matrix[index1][index2] = act(*data)
            else:
                raise TypeError('matrices of different sizes cannot be multiplied in this way')
        return new_matrix
    def __mul__(self, other):
        from operator import mul
        return self.template(other, mul)
    def __add__(self, other):
        from operator import add
        return self.template(other, add)
    def __iadd__(self, other):
        from operator import add
        return self.template(other, add)
    def __radd__(self, other):
        from operator import add
        return self.template(other, add, reverse=True) 
    def __sub__(self, other):
        from operator import sub
        return self.template(other, sub)
    def __rtruediv__(self, other):
        from operator import truediv
        return self.template(other, truediv, reverse=True)
    def __rsub__(self, other):
        from operator import sub
        return self.template(other, sub, reverse=True)
    def __rpow__(self, other):
        return self.template(other, pow, reverse=True)
    @property
    def coup(self):
        new_matrix = Matrix([[] for i in range(len(self[0]))])
        for index1 in range(len(self[0])):
            new_matrix[index1].extend([i[index1] for i in self])
        return new_matrix

def Z(inputs, weights):
    new_matrix = Matrix([])
    for index1, value1 in enumerate(inputs):
        result = 0
        for index2, value2 in enumerate(value1):
            result += value2 * weights[index2][0]
        new_matrix.append([result])
    return new_matrix

def adjust_the_weights(coup_input, intermediate_result):
    new_matrix = Matrix([])
    for index1, value1 in enumerate(coup_input):
        result = 0
        for index2, value2 in enumerate(value1):
            result += value2 * intermediate_result[index2][0]
        new_matrix.append([result])
    return new_matrix


training_set_inputs = Matrix([  [0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

training_set_outputs = Matrix([ [0],
                                [1],
                                [1],
                                [0]])

synaptic_weights = Matrix([[uniform(-1.0, 1.0)] for i in range(3)])

def sigmoid(x):
    return 1 / (1 + (e ** -x))

output = sigmoid(Z(training_set_inputs, synaptic_weights))

for i in range(20000):
    synaptic_weights += adjust_the_weights(training_set_inputs.coup, training_set_outputs - output * (1 - output))

new_input = Matrix([[1, 0, 0]])
print(sigmoid(Z(new_input, synaptic_weights)))
