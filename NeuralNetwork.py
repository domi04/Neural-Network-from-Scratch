import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:

    def __init__(self,n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculating output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
 
    def forward(self, inputs):
        # Calculating output values from inputs using the activation function ReLU 
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero Gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Raising Eulers number e to the inputs value and subtracting the max
                                                                            # value to prevent the output from being very large values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalizing the values  
        self.output = probabilities

class Loss:
    def calculate(self,output,y):

        sample_loss = self.forward(output,y) # Sample Losses

        data_loss = np.mean(sample_loss) # Mean Loss

        return data_loss

class Loss_CatgeoricalCrossentropy(Loss):
    def forward(self,y_pred, y_true):
        
        samples = len(y_pred)

        #C lip data to prevent divsion by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true]

        # Probabilities for one-hot encoded target values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelyhoods = - np.log(correct_confidences)
        return negative_log_likelyhoods

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)

activation2 = Activation_Softmax()

loss_function = Loss_CatgeoricalCrossentropy()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)

# Accuracy Calculation
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy_score = np.mean (predictions==y)

print(f"Accuracy:{accuracy_score}")
print(f"Loss:{loss}")
