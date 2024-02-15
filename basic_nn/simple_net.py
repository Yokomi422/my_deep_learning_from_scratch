import numpy as np
import pickle
import sys,os

sys.path.append("..")
from common.functions import sigmoid,softmax,cross_entropy
from common.gradient import numerical_gradient

save_file = "trained_params.pkl"


class SimpleNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std: float = 0.01):
        self.params = {}

        if os.path.exists(save_file):
            with open(save_file,'rb') as f:
                self.params = pickle.load(f)
        else:
            self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
            self.params["b1"] = weight_init_std * np.zeros(hidden_size)
            self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
            self.params["b2"] = weight_init_std * np.zeros(output_size)

    def predict(self,x):
        x_flatten = x.reshape(x.shape[0],-1)
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x_flatten,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y

    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy(y,t)

    def calculate_numerical_gradient(self,x,t):
        # W1やb1でどうして偏微分ができるのか。 predictのなかにW1やb1が入っていて、それを微小変化させている
        # paramsの値が微小変化している
        loss_W = lambda W: self.loss(x,t)
        # どうして、tとxを固定していて、W1で偏微分をできるのかがわからない
        grads = {'W1': numerical_gradient(loss_W,self.params['W1']),'b1': numerical_gradient(loss_W,self.params['b1']),
                 'W2': numerical_gradient(loss_W,self.params['W2']),'b2': numerical_gradient(loss_W,self.params['b2'])}

        return grads
