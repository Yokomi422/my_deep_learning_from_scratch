import numpy as np
import pickle
import sys,os

sys.path.append("..")
from common.functions import sigmoid,softmax,cross_entropy
from common.gradient import numerical_gradient


class SimpleNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std: float = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = weight_init_std * np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = weight_init_std * np.zeros(output_size)

    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y

    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy(y,t)

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        # どうして、tとxを固定していて、W1で偏微分をできるのかがわからない
        grads = {'W1': numerical_gradient(loss_W,self.params['W1']),'b1': numerical_gradient(loss_W,self.params['b1']),
                 'W2': numerical_gradient(loss_W,self.params['W2']),'b2': numerical_gradient(loss_W,self.params['b2'])}

        return grads