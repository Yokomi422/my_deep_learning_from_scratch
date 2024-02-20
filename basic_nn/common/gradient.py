import numpy as np


def _gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        upper = f(x)

        x[idx] = tmp - h
        lower = f(x)

        grad[idx] = (upper + lower) / (2 * h)
        x[idx] = tmp

    return grad


# 多次元の場合も考慮した勾配法の実装
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _gradient_1d(f, X)
    grad = np.zeros_like(X)

    for idx, x in enumerate(X):
        grad[idx] = _gradient_1d(f, x)

    return grad
