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

    def backward(self, dvalues):

        # create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1,1)
            # calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # calculate sample-wise gradient 
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues) 

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

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # number of labels in every sample
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector 
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    # creates activation and loss function objects
    self.activation = Activation_Softmax()
    self.loss = Loss_CatgeoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
    
        # if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy to modify safely
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / samples

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
