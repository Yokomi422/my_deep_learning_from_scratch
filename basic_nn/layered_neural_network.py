import sys, os
import numpy as np

sys.path.append(os.pardir)

from data.cifar import load_cifar
from basic_nn.common.layered_network import TwoLayerNet

"""
第5章の逆誤差伝播法を用いたニューラルネットワークの実装
"""
input_size = 3 * 32 * 32
hidden_size = 100
output_size = 10
weight_init_std = 0.01
learning_rate = 0.01
# エポックは120ページを参照
epochs = 100
# バッチサイズは79ページ参照
batch_size = 100

(x_train,t_train),(x_test,t_test) = load_cifar(normalize=True,one_hot_label=True,flatten=True)

# ネットワークの初期化 keyを書かないと、setとみなされるので注意する
params = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "weight_init_std": 0.01,
}
network = TwoLayerNet(params=params)
iter_nums = 1000
train_size = x_train.shape[0]

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iter_nums):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)
