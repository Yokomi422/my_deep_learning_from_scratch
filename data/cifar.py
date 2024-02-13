import os
import pickle
import requests
import tarfile
import numpy as np

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dataset_dir = "cifar-10-batches-py"
save_file = "cifar-10-python.pkl"

key_files = {
    'train': "data_batch_1",
    'test': 'test_batch',
}


def _download():
    response = requests.get(url)
    # cifar-10-python.tar.gzに書き込む
    with open("cifar-10-python.tar.gz","wb") as f:
        f.write(response.content)
    with tarfile.open("cifar-10-python.tar.gz","r:gz") as tar:
        tar.extractall()
    os.remove("cifar-10-python.tar.gz")


def load_cifar_batch(file_name):
    with open(file_name,'rb') as file:
        # pickle形式で保存されているので、pickleで読み込む
        batch = pickle.load(file,encoding='latin1')
        data = batch['data']
        labels = batch['labels']
        # 画像が10000個あって、rgbの3チャンネル、32x32の画像
        data = data.reshape((10000,3,32,32)).astype('float32')
        return data,labels


def _convert_numpy():
    dataset = {}
    for key in ('train','test'):
        data,labels = load_cifar_batch(os.path.join(dataset_dir,key_files[key]))
        if key == 'train':
            dataset['train_img'] = data
            dataset['train_label'] = labels
        else:
            dataset['test_img'] = data
            dataset['test_label'] = labels
    return dataset


def init_cifar():
    _download()
    dataset = _convert_numpy()
    with open(save_file,'wb') as f:
        pickle.dump(dataset,f,-1)


def _convert_labels_to_one_hot(labels,num_classes = 10):
    return np.eye(num_classes)[labels]


def load_cifar(normalize = True,one_hot_label = True, flatten = True):
    if not os.path.exists(save_file):
        init_cifar()

    with open(save_file,'rb') as f:
        dataset = pickle.load(f)

    # 画像のピクセル値を0.0~1.0に正規化
    if normalize:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset['train_label'] = _convert_labels_to_one_hot(dataset['train_label'])
        dataset['test_label'] = _convert_labels_to_one_hot(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 32, 32)

    return (dataset['train_img'],dataset['train_label']),(dataset['test_img'],dataset['test_label'])
