import numpy as np
import os,sys

sys.path.append("..")

from data.cifar import load_cifar

(train_images,train_labels),(test_images,test_labels) = load_cifar()


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
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return - np.sum(np.log(y[np.arange(batch_size), t])) / batch_size





t = np.array([[0,0,1,0,0,0,0,0,0,0],[2,4,1,4,1,51,5,1,5,2]])
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])


y = y.reshape(1, y.size)
print(y.shape[0])
