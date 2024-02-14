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
        return - np.sum(t * np.log(y))
