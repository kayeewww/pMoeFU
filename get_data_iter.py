import os
import sys
import torch
import pickle
import numpy as np
from sklearn.utils import shuffle
from torchvision.datasets import MNIST
from torchvision import transforms

IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


# Cutout 是一种常用的数据增强方法，通过在图像中随机遮挡一块区域来增强模型的鲁棒性和泛化能力。
# 可选参数 length 表示遮挡区域的长度
def cutout_func(img, length=16):
    h, w = img.size(1), img.size(2)

    mask = np.ones((h, w), np.float32)
    #创建一个与输入图像相同大小的全 1 数组作为遮挡的掩码

    y = np.random.randint(h)
    x = np.random.randint(w)
    #随机生成遮挡区域的左上角顶点坐标。

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    #计算遮挡区域的左上角顶点坐标和右下角顶点坐标，并通过 np.clip 函数确保坐标不超出图像边界

    mask[y1: y2, x1: x2] = 0.
    #将遮挡区域的像素值设为 0，实现遮挡效果

    # mask = torch.from_numpy(mask)
    mask = mask.reshape(img.shape)
    img *= mask
    return img

#可以同时处理多张图片
def cutout_batch(img, length=16):
    h, w = img.size(2), img.size(3)
    masks = [] #用于存储每张图像的遮挡掩码
    for i in range(img.size(0)):
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img[0]).unsqueeze(0)
        masks.append(mask)
    # masks = torch.cat(masks).cuda()
    masks = torch.cat(masks)
    #将 masks 列表中的掩码张量拼接成一个张量，以便与输入图像张量进行相乘
    img *= masks
    return img


class CIFAR_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, data_type='train', root='/userhome/data/cifar10'):
        self.root = root
        self.batch_size = batch_size
        self.train = (data_type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        np.save("cifar.npy", self.data)
        self.data = np.load('cifar.npy')  # to serialize, increase locality

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(
                    self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__

class MNIST_INPUT_ITER():
    base_folder = 'MNIST'
    train_list = [
        ['train-images-idx3-ubyte', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'],
        ['train-labels-idx1-ubyte', 'd53e105ee54ea40749a09fcbcd1e9432'],
    ]

    test_list = [
        ['t10k-images-idx3-ubyte', '9fb629c4189551a2d022fa330f9573f3'],
        ['t10k-labels-idx1-ubyte', 'ec29112dd5afa0611ce80d1b7f02629c'],
    ]

    def __init__(self, batch_size, data_type='train', root='./data'):
        self.root = root
        self.batch_size = batch_size
        self.train = (data_type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        transform = transforms.ToTensor()
        mnist_data = MNIST(root=root, train=self.train, download=True, transform=transform)

        for i in range(len(mnist_data)):
            img, label = mnist_data[i]
            self.data.append(img.numpy())
            self.targets.append(label)

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__

