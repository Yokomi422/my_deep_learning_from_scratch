import os,sys
from pydantic import BaseModel,Field
import numpy as np
# OrderedDictは追加した順番を保持する辞書
from collections import OrderedDict

from .layers import AffineLayer,ReLuLayer,SigmoidLayer,SoftmaxWithLossLayer
from .gradient import numerical_gradient


class TwoLyNNParams(BaseModel):
    input_size: int = Field(...,gt=1)
    hidden_size: int = Field(...,gt=1)
    output_size: int = Field(...,gt=1)
    weight_init_std: float = Field(default=0.01,gt=0)


sys.path.append(os.pardir)


class TwoLayerNet:
    def __init__(self,params: TwoLyNNParams):
        self.params = {
            "W1": params["weight_init_std"] * np.random.randn(params["input_size"],params["hidden_size"]),
            "b1": params["weight_init_std"] * np.zeros(params["hidden_size"]),
            "W2": params["weight_init_std"] * np.random.randn(params["hidden_size"],params["output_size"]),
            "b2": params["weight_init_std"] * np.zeros(params["output_size"])
        }
        # レイヤーの作成
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = ReLuLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'],self.params['b2'])

        self.lastLayer = SoftmaxWithLossLayer()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1: t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {'W1': numerical_gradient(loss_W,self.params['W1']),'b1': numerical_gradient(loss_W,self.params['b1']),
                 'W2': numerical_gradient(loss_W,self.params['W2']),'b2': numerical_gradient(loss_W,self.params['b2'])}

        return grads

    def gradient(self,x,t):
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {'W1': self.layers['Affine1'].dW,'b1': self.layers['Affine1'].db,'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db}

        return grads
