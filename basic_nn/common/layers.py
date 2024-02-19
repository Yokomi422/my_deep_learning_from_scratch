import numpy as np
from pydantic import BaseModel,Field


class ReLu:
    """
    x <= 0なら0, x > 0ならxを出力する活性化関数レイヤー
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

