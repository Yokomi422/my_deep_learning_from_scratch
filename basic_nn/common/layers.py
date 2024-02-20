import sys

import numpy as np
from pydantic import BaseModel, Field

from basic_nn.common.functions import cross_entropy, softmax


class ReLuLayer:
    """
    x <= 0なら0, x > 0ならxを出力する活性化関数レイヤー
    xは多次元でも対応できる
    """

    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray):
        """
        順伝播を行う

        Parameters:
            - x: np.ndarray: 入力データ
        Returns:
            - np.ndarray: 出力データ
        """
        self.mask = x <= 0
        # 元のxをメモリに保持したまま、元のxと別に他の参照を作るためにcopy
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray):
        """
        逆伝播を行う
        Parameters:
            - dout: np.ndarray: 上流からの勾配
        Returns:
            - np.ndarray: 下流への勾配
        """
        # こちらでcopyしないのは、もうdoutを計算に使用しないから
        dout[self.mask] = 0
        dx = dout
        return dx


class SigmoidLayer:
    """
    Sigmoid関数の活性化レイヤー
    """

    def __init__(self):
        self.out: float

    def forward(self, x: np.ndarray):
        """
        順伝播を行う
        Parameters:
            x: np.ndarray: 入力データ
        Returns:
            np.ndarray: 出力データ
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout: np.ndarray):
        """
        逆伝播を行う
        Parameters:
            dout: np.ndarray: 上流からの勾配
        Returns:
            np.ndarray: 下流への勾配
        """
        dx = dout * (1.0 - self.out) * self.out

        return dx


class AffineLayer:
    """
    入力データの行列Xと重みWの行列積のレイヤー
    """

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.original_x_shape: tuple[int, ...] | None = None
        self.W: np.ndarray = W
        self.b: np.ndarray = b

        self.x: np.ndarray
        # 重み・バイアスパラメータの微分
        self.dW: np.ndarray
        self.db: np.ndarray

    def forward(self, x: np.ndarray):
        """
        順伝播を行う
        Parameters:
            x: np.ndarray: 入力データ
        Returns:
            np.ndarray: 出力データ
        """
        # テンソルにも対応しているらしいが、わからず
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = x @ self.W + self.b

        return out

    def backward(self, dout: np.ndarray):
        """
        逆伝播を行う
        Parameters:
            dout: np.ndarray: 上流からの勾配
        Returns:
            np.ndarray: 下流への勾配
        """
        # dxは次のノードに渡す値なので、保持する必要がない
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(
            *self.original_x_shape
        )  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLossLayer:
    """
    導出は付録を参照
    """

    def __init__(self):
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
