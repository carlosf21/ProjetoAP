#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1.0, 0.0)


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        # Subtrai o max de cada linha para estabilidade numérica (evita overflow)
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward_propagation(self, output_error):
        # Override necessário porque a derivada do Softmax em relação ao input
        # interage com todo o array de saída (matriz Jacobiana), não apenas elemento a elemento.
        A = self.output
        return A * (output_error - np.sum(output_error * A, axis=1, keepdims=True))

    def derivative(self, input):
        # Não é usado diretamente devido ao override do backward_propagation acima.
        pass