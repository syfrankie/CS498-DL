"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        #self.w = np.zeros((self.n_class, len(X_train[0])))
        self.w = np.random.normal(0, .05, size=(self.n_class, len(X_train[0])))
        for _ in range(self.epochs):
            for i in range(X_train.shape[0]):
                pred = np.dot(X_train[i], self.w.T)
                if np.argmax(pred) == y_train[i]:
                    continue
                for j in range(self.n_class):
                    if pred[j] > pred[y_train[i]]:
                        #temp = pred[j] - pred[y_train[i]]
                        self.w[j, :] = np.subtract(self.w[j, :], self.lr * X_train[i, :])
                        self.w[y_train[i], :] = np.add(self.w[y_train[i], :], self.lr * X_train[i, :])
                #self.w[y_train[i]] = np.add(self.w[y_train[i]], self.lr * gradient * X_train[i])



    def predict(self, X_test):
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        out = []
        pred = np.dot(X_test, self.w.T)
        for i in range(X_test.shape[0]):
            out.append(np.argmax(pred[i]))
        return out
