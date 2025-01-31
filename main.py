import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Loss:
      def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss): # inherits Loss class to help self calculate losses
    def forward(self, y_pred, y_true):
        # y_pred is the output of the model, y_true is the true label
        # y_pred is a 2D array with shape (batch_size, num_classes)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Activation_Softmax: # softmax algorithim for classifying, use on output layer
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Activation_ReLU: # ReLU algorithm for sorting data
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # randomly apply weights to the neurons
        self.biases = np.zeros((1, n_neurons)) # initialize biases to 0
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # move through the layer and apply the weights and biases

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense1.forward(X)

activation1.forward(dense1.output)

print(activation1.output[:5])