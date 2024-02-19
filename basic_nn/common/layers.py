import numpy as np
from pydantic import BaseModel,Field


class ReLuLayer:
    """
    x <= 0なら0, x > 0ならxを出力する活性化関数レイヤー
    xは多次元でも対応できる
    """

    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self,x: np.ndarray):
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

    def backward(self,dout: np.ndarray):
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
        self.out: float | None = None

    def forward(self,x: np.ndarray):
        """
        順伝播を行う
        Parameters:
            x: np.ndarray: 入力データ
        Returns:
            np.ndarray: 出力データ
        """
        out = 1 / (1 + np.exp(- x))
        self.out = out

        return out

    def backward(self,dout: np.ndarray):
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
    def __init__(self,W: np.ndarray,b: np.ndarray):
        self.W: np.ndarray | None = W
        self.b: np.ndarray | None = b

        self.x: np.ndarray | None = None
        self.original_x_shape: np.ndarray | None = None
        # 重み・バイアスパラメータの微分
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self,x: np.ndarray):
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

    def backward(self,dout: np.ndarray):
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

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

