import numpy as np
import matplotlib.pyplot as plt

from cifar import load_cifar

# CIFAR-10データセットの読み込み
(train_images, train_labels), (test_images, test_labels) = load_cifar(normalize=False, flatten=False)

import matplotlib.pyplot as plt
images = train_images.reshape(-1,3,32,32).transpose(0,2,3,1)

idx = np.random.randint(images.shape[0]) # ランダムにインデックスを選択
selected_image = images[idx]
plt.imshow(selected_image)
plt.title(f'Label: {train_labels[idx]}')
plt.show()
