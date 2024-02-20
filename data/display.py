import matplotlib.pyplot as plt
import numpy as np

from data.cifar import load_cifar

(train_images, train_labels), (test_images, test_labels) = load_cifar(
    normalize=False, flatten=False
)

import matplotlib.pyplot as plt

# transposeは配列の順番を変えるためにする 一般的なライブラリでは、(縦, 横, channel size)になっているので合わせる
images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
idx = np.random.randint(images.shape[0])
selected_image = images[idx]
plt.imshow(selected_image)
plt.title(f"Label: {train_labels[idx]}")
plt.show()
