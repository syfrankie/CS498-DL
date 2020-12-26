"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.add(np.dot(X, W), b)

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(0, X)

    def relu_grad(self, X):
        """
        Derivative of ReLU
        """
        X[X<=0] = 0
        X[X>0] = 1
        return X

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.divide(np.exp(X), np.sum(np.exp(X), axis=1, keepdims=True))

    def crossentropy(self, label, pred):
        ce = -np.sum(np.multiply(label, np.log(pred))) / pred.shape[0]
        return ce

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        z = self.linear(self.params['W' + str(1)], X, self.params['b' + str(1)])
        self.outputs["z" + str(1)] = z
        for i in range(2, self.num_layers + 1):
            z = self.relu(z)
            z = self.linear(self.params['W' + str(i)], z, self.params['b' + str(i)])
            self.outputs["z" + str(i)] = z

        return self.softmax(z)

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        #loss = 0.0
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        scores = self.softmax(self.outputs["z" + str(self.num_layers)])
        labels = np.zeros(scores.shape)
        for i in range(len(y)):
            labels[i, y[i]] = 1
        loss = self.crossentropy(labels, scores)

        # rho_p is the derivative of loss
        rho_p = -np.subtract(labels, scores)
        delta = np.dot(self.params["W" + str(self.num_layers)], rho_p.T)
        self.gradients["W" + str(self.num_layers)] = np.dot(self.relu(self.outputs["z" + str(self.num_layers - 1)]).T, rho_p)
        self.gradients["b" + str(self.num_layers)] = np.mean(rho_p, axis=0, keepdims=False)

        for i in range(self.num_layers - 1, 0, -1):
            if i == 1:
                temp = X
            else:
                temp = self.relu(self.outputs['z' + str(i - 1)])
            self.gradients["W" + str(i)] = np.dot(temp.T, np.multiply(delta, self.relu_grad(self.outputs["z" + str(i)]).T).T)
            self.gradients["b" + str(i)] = np.mean(np.multiply(delta, self.relu_grad(self.outputs["z" + str(i)]).T), axis=1, keepdims=False)
            delta = np.dot(self.params["W" + str(i)], np.multiply(delta, self.relu_grad(self.outputs["z" + str(i)]).T))

        return loss
