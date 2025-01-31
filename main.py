import numpy as np
import math
import nnfs
from nnfs.datasets import vertical_data

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

X,y = vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3) # 2 inputs, 3 neurons

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3) # 3 neurons, 3 outputs

activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = 9999999  # some initial value 
best_dense1_weights = dense1.weights.copy() 
best_dense1_biases = dense1.biases.copy() 
best_dense2_weights = dense2.weights.copy() 
best_dense2_biases = dense2.biases.copy() 
 
for iteration in range(10000): 
 
    # Update weights with some small random values 
    dense1.weights += 0.05 * np.random.randn(2, 3) 
    dense1.biases += 0.05 * np.random.randn(1, 3) 
    dense2.weights += 0.05 * np.random.randn(3, 3) 
    dense2.biases += 0.05 * np.random.randn(1, 3) 
 
    # Perform a forward pass of our training data through this layer 
    dense1.forward(X) 
    activation1.forward(dense1.output) 
    dense2.forward(activation1.output) 
    activation2.forward(dense2.output) 
 
    # Perform a forward pass through activation function 
    # it takes the output of second dense layer here and returns loss 
    loss = loss_function.calculate(activation2.output, y)  
    # Calculate accuracy from output of activation2 and targets 
    # calculate values along first axis 
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y) 
 
    # If loss is smaller - print and save weights and biases aside 
    if loss < lowest_loss: 
        print('New set of weights found, iteration:', iteration, 
              'loss:', loss, 'acc:', accuracy) 
        best_dense1_weights = dense1.weights.copy() 
        best_dense1_biases = dense1.biases.copy() 
        best_dense2_weights = dense2.weights.copy() 
        best_dense2_biases = dense2.biases.copy() 
        lowest_loss = loss 
    # Revert weights and biases 
    else: 
        dense1.weights = best_dense1_weights.copy() 
        dense1.biases = best_dense1_biases.copy() 
        dense2.weights = best_dense2_weights.copy() 
        dense2.biases = best_dense2_biases.copy()