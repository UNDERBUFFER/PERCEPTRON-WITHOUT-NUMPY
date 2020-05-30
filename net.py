from numpy import array, dot
from random import uniform
from math import e

my_training_set_inputs = array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

my_training_set_outputs = array([   [0],
                                    [1],
                                    [1],
                                    [0]])

my_synaptic_weights = array([[uniform(-1.0, 1.0)] for i in range(3)])

def sigmoid(x):
    return 1 / (1 + (e ** -x))

my_output = sigmoid(dot(my_training_set_inputs, my_synaptic_weights))

for i in range(100):
    my_synaptic_weights += dot(my_training_set_inputs.T, my_training_set_outputs - my_output * (1 - my_output))

new_input = array([[1, 0, 0]])


print(sigmoid(dot(new_input, my_synaptic_weights)))

class Matrix(list):
    def __repr__(self):
        return 'Matrix({})'.format(super().__repr__())
    def __neg__(self):
        for index1, value1 in enumerate(self):
            for index2, value2 in enumerate(value1):
                self[index1][index2] = -value2
        return self
    def __mul__(self, other):
        if isinstance(other, int):
            for index1, value1 in enumerate(self):
                for index2, value2 in enumerate(value1):
                    self[index1][index2] = value2 * other
        else:
            if (len(self), len(self[0])) == (len(other), len(other[0])):
                for index1, value1 in enumerate(self):
                    for index2, value2 in enumerate(value1):
                        self[index1][index2] = value2 * other[index1][index2]
            else:
                raise TypeError('matrices of different sizes cannot be multiplied in this way')
        return self
    def __add__(self, other):
        if isinstance(other, int):
            for index1, value1 in enumerate(self):
                for index2, value2 in enumerate(value1):
                    self[index1][index2] = value2 + other
        else:
            if (len(self), len(self[0])) == (len(other), len(other[0])):
                for index1, value1 in enumerate(self):
                    for index2, value2 in enumerate(value1):
                        self[index1][index2] = value2 + other[index1][index2]
            else:
                raise TypeError('matrices of different sizes cannot be multiplied in this way')
        return self
    def __rdiv__(self, other):
        if isinstance(other, int):
            for index1, value1 in enumerate(self):
                for index2, value2 in enumerate(value1):
                    self[index1][index2] = other / value2
        else:
            if (len(self), len(self[0])) == (len(other), len(other[0])):
                for index1, value1 in enumerate(self):
                    for index2, value2 in enumerate(value1):
                        self[index1][index2] = other[index1][index2] / value2
            else:
                raise TypeError('matrices of different sizes cannot be multiplied in this way')
        return self
    def __rsub__(self, other):
        if isinstance(other, int):
            for index1, value1 in enumerate(self):
                for index2, value2 in enumerate(value1):
                    self[index1][index2] = other - value2
        else:
            if (len(self), len(self[0])) == (len(other), len(other[0])):
                for index1, value1 in enumerate(self):
                    for index2, value2 in enumerate(value1):
                        self[index1][index2] = other[index1][index2] - value2
            else:
                raise TypeError('matrices of different sizes cannot be multiplied in this way')
        return self
    def __sub__(self, other):
        return super.__rsub__(other, self)
    @property
    def coup(self):
        new_matrix = Matrix([[] for i in range(len(self[0]))])
        for index1 in range(len(self[0])):
            new_matrix[index1].extend([i[index1] for i in self])
        return new_matrix