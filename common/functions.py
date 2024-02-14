import numpy as np
import os,sys

sys.path.append("..")


# 出力層の関数
def softmax(a: np.ndarray):
    # それぞれのデータから最大値を取り出す
    c = np.max(a, axis=-1, keepdims=True)
    # オーバーフロー対策と指数関数の特性を利用して、aの最大値を引いている
    # aを変更するのは良くない
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


def cross_entropy(y: np.ndarray,t: np.ndarray) -> float:
    """
    Parameters:
        y: 予測結果が[0, 1]の範囲で格納されたnp.ndarray
        t: 正解データがone-hot-encodingされたnp.ndarray

    Returns:
        交差エントロピー誤差: float
    """
    # logに0が入らないように微小な値を代入する
    h = 1e-4
    y += h
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return - np.sum(np.log(y[np.arange(batch_size),t])) / batch_size
