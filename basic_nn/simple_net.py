import numpy as np
import pickle
import sys,os

sys.path.append("..")
from common.functions import sigmoid,softmax


class SimpleNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.params = {}
        self.params["W1"] = np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y