import sys

sys.path.append("..")
import pickle

import numpy as np

from basic_nn.common.simple_net import SimpleNeuralNetwork
from data.cifar import load_cifar

save_file = "trained_params.pkl"

## TODO input_sizeとパラメータ数が多くて現実的な計算では終わらない.
input_size = 3 * 32 * 32
hidden_size = 100
output_size = 10
learning_rate = 0.01
# エポックは120ページを参照
epochs = 100
# バッチサイズは79ページ参照
batch_size = 100

(x_train, t_train), (x_test, t_test) = load_cifar(
    normalize=True, one_hot_label=True, flatten=True
)

# ニューラルネットワークの初期化
network = SimpleNeuralNetwork(input_size, hidden_size, output_size)

for epoch in range(epochs):
    # np.random.choice(x, y)は0からx-1の数字からランダムでy個の配列を重複ありで生成
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.calculate_numerical_gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grads[key]

    # 学習進捗の表示
    loss = network.loss(x_batch, t_batch)
    print("Epoch: {}, Loss: {}".format(epoch + 1, loss))

with open(save_file, "wb") as f:
    pickle.dump(network.params, f)
