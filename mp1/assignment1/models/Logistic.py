"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z):
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.random.normal(0, .05, size=(np.shape(X_train)[1], 1))
        new_y = np.zeros((y_train.shape))
        for j in range(y_train.shape[0]):
            if y_train[j] == 0:
                new_y[j] = -1
            else:
                new_y[j] = 1
        for _ in range(self.epochs):
            pred = self.sigmoid(np.multiply(-1 * new_y.reshape(X_train.shape[0], 1), np.dot(X_train, self.w)))
            temp = np.zeros((X_train.shape))
            for i in range(X_train.shape[0]):
                temp[i, :] = pred[i] * new_y[i] * X_train[i, :]
            self.w = np.add(self.w, self.lr * np.expand_dims(np.mean(temp, axis=0), axis=1))
            
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        pred = self.sigmoid(np.dot(X_test, self.w))
        for i in range(X_test.shape[0]):
            if pred[i] >= self.threshold:
                pred[i] = 1
            else:
                pred[i] = 0
        return pred
