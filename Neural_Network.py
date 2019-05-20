import numpy as np
from utils import *

class NeuralNetwork:

    def __init__(self, x, y, hidden_size):
    	self.hidden_size= hidden_size
    	self.input = x
    	#print(np.shape(x))
    	self.weights1 = np.random.rand((np.shape(self.input))[1],hidden_size) 
    	self.weights2 = np.random.rand(hidden_size,np.shape(y)[1])                 
    	self.y = y
    	#print(np.shape(y))
    	self.output = np.zeros([y.shape[0], y.shape[1]])

    def feedforward(self,input):
    	self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    	self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def train(self, number_of_iterations):
    	for iter in range(number_of_iterations):
    		self.feedforward(self.input)
    		self.backprop()

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        #print("output")
        #print(self.output)
        #print("sigmoid_derivative")
        #print(sigmoid_derivative(self.output))
        int1 = 2*(self.y - self.output) * sigmoid_derivative(self.output)
        #print(int1)
        int2 = np.dot(int1, self.weights2.T)
        int3 = int2 * sigmoid_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T,  int2)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2