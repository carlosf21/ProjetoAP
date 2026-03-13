#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from layers import DenseLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from metrics import mse


class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = MeanSquaredError,
                 metric:callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        self.verbose = verbose
        # ALTERAÇÃO: Permite passar uma Loss já parametrizada (ex: com Class Weights)
        self.loss = loss if not isinstance(loss, type) else loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None, shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        # COMPLETADO: O erro tem de propagar-se de trás para a frente (reverse)
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        
        # ALTERAÇÃO: Se for multi-classe (y já for uma matriz one-hot), não fazemos expand_dims.
        # Só fazemos se y for um vetor simples (1 dimensão)
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            
            # Arrays para guardar todas as previsões e labels do epoch (para calcular a métrica final)
            all_predictions = []
            all_true_labels = []
            
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # 1. Forward propagation no mini-batch
                output = self.forward_propagation(X_batch, training=True)
                
                # 2. Calcular o erro (derivada da loss) da rede em relação a este mini-batch
                error = self.loss.derivative(y_batch, output)
                
                # 3. Backward propagation no mini-batch (atualiza os pesos!)
                self.backward_propagation(error)

                all_predictions.append(output)
                all_true_labels.append(y_batch)

            # Juntar as previsões de todos os mini-batches
            output_x_all = np.concatenate(all_predictions)
            y_all = np.concatenate(all_true_labels)

            # Calcular a Loss total no final da Época
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # Guardar histórico
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")