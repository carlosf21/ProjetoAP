#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)


class BinaryCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Dividimos pelo tamanho do batch para calcular a média do gradiente
        return ((1 - y_true) / (1 - p) - y_true / p) / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    
    def __init__(self, class_weights=None):
        """
        class_weights: lista ou array numpy com os pesos de cada classe (para lidar com dados desbalanceados).
        """
        self.class_weights = class_weights

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if self.class_weights is not None:
            weights = np.array(self.class_weights)
            # Aplica o peso específico da classe real de cada amostra
            pesos_amostras = np.sum(y_true * weights, axis=1)
            loss_amostras = -np.sum(y_true * np.log(p), axis=1) * pesos_amostras
            return np.mean(loss_amostras)
        else:
            return -np.mean(np.sum(y_true * np.log(p), axis=1))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        m = y_true.shape[0] # Tamanho do batch
        
        if self.class_weights is not None:
            weights = np.array(self.class_weights)
            pesos_amostras = np.sum(y_true * weights, axis=1, keepdims=True)
            # O gradiente também é multiplicado pelo peso da classe
            return - (y_true / p) * pesos_amostras / m
        else:
            return - (y_true / p) / m