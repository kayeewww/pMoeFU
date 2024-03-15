from collections import OrderedDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,TensorDataset
from torchvision import datasets, transforms
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.sparse import load_npz

from get_data_iter import cutout_batch


def model_init(data_name, model_name):
    if (data_name == 'cifar10'):
        return Net_cifar10(model_name)
    elif (data_name == 'cifar100'):
        return Net_cifar100(model_name)
    elif (data_name == 'mnist'):
        return Net_mnist(model_name)
    elif(data_name == 'purchase'):
        return Net_purchase(model_name)
    elif(data_name == 'adult'):
        return Net_adult(model_name)

"""Function: load data"""


def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)
    # shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
    # trainset, shadow _trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)

    # shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
    # testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)
    #

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
    # shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
    split_index = [int(trainset.__len__() / FL_params.N_total_client)] * (FL_params.N_total_client - 1)
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_params.N_total_client) * (FL_params.N_total_client - 1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    # split_index = [int(shadow_trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    # split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    # shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
    # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
    client_loaders = []
    # shadow_client_sloaders = []
    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
        # shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        '''
        By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
        '''

    return client_loaders, test_loader

def data_set(FL_params,data_name):
    if not data_name in ['mnist', 'purchase', 'adult', 'cifar10', 'cifar100']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    if FL_params.data_name == 'mnist':
        trainset = datasets.MNIST('../data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.MNIST('../data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    elif FL_params.data_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='../data', train=True,
                                    download=True, transform=transform)

        testset = datasets.CIFAR10(root='../data', train=False,
                                   download=True, transform=transform)
    elif FL_params.data_name == 'adult':
        # load data
        file_path = "./data/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None, skiprows=1)
        data2 = data2.replace(' <=50K.', ' <=50K')
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        data = pd.concat([data1, data2])

        # data transform: str->int
        data = np.array(data, dtype=str)
        labels = data[:, 14]
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, :-1]

        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)

        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:, feature].reshape(-1, 1))
            data[:, feature] = sacled_data.reshape(-1)

        # OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse_output=False), categorical_features), ],
            remainder='passthrough')
        oh_data = oh_encoder.fit_transform(data)

        xx = oh_data
        yy = labels
        # 最终处理，xx进行规范化
        xx = preprocessing.scale(xx)
        yy = np.array(yy)

        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0], :]
        xx_test = xx[data1.shape[0]:, :]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]

        trainset = TensorDataset(xx_train, yy_train)
        testset = TensorDataset(xx_test, yy_test)
    elif FL_params.data_name == 'purchase':
        data = np.concatenate(
            [load_npz("./data/purchase/data1.npz").toarray(), load_npz("./data/purchase/data2.npz").toarray()]).astype(
            int)
        num_class = 2
        if not os.path.exists(f"./data/purchase/{num_class}_kmeans.npy"):
            kmeans = KMeans(n_clusters=num_class, random_state=0).fit(data)
            label = kmeans.labels_
            np.save(f"./data/purchase/{num_class}_kmeans.npy", label)
        else:
            label = np.load(f"./data/purchase/{num_class}_kmeans.npy")

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)
    return trainset, testset


"""Function: load data"""
# 影子数据加载器的目的是为了在联邦学习中防止过拟合，因为它们不参与训练，只用于评估模型的性能
def data_init_with_shadow(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    whole_trainset, whole_testset = data_init(FL_params.data_name)
    shadow_split_idx = [int(whole_trainset.__len__() / 2),
                        int(whole_trainset.__len__()) - int(whole_trainset.__len__() / 2)]
    trainset, shadow_trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)

    shadow_split_idx = [int(whole_testset.__len__() / 2),
                        int(whole_testset.__len__()) - int(whole_testset.__len__() / 2)]
    testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
    shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
    split_index = [int(trainset.__len__() / FL_params.N_client)] * (FL_params.N_client - 1)
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_params.N_client) * (FL_params.N_client - 1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    split_index = [int(shadow_trainset.__len__() / FL_params.N_client)] * (FL_params.N_client - 1)
    split_index.append(
        int(shadow_trainset.__len__() - int(shadow_trainset.__len__() / FL_params.N_client) * (FL_params.N_client - 1)))
    shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
    # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
    client_loaders = []
    shadow_client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        shadow_client_loaders.append(
            DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        '''
        By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
        '''

    return client_loaders, test_loader, shadow_client_loaders, shadow_test_loader

# define class->dataset  for adult and purchase datasets
# for the purchase, we use TensorDataset function to transform numpy.array to datasets class
# for the adult, we custom an AdultDataset class that inherits torch.util.data.Dataset class
"""
Array2Dataset: A class that can transform np.array(tensor matrix) to a torch.Dataset class.  
"""
class Array2Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)

        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=True, in_channels=3):
        super(ResNet, self).__init__()
        if cfg is None:
            cfg = [16, 16, 32, 64]
        num_blocks = []
        if depth == 20:
            num_blocks = [3, 3, 3]
        elif depth == 32:
            num_blocks = [5, 5, 5]
        elif depth == 44:
            num_blocks = [7, 7, 7]
        elif depth == 56:
            num_blocks = [9, 9, 9]
        elif depth == 110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16 #当前模块的输入通道数，也即上一个模块的输出通道数
        self.in_channels = in_channels
        conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.layer1 = self._make_layer(block, cfg[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[3], num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d' % i, block(self.in_planes, planes, strides[i])))
            self.in_planes = planes
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Net_cifar10(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=10, in_channels=3)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=10, in_channels=3)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=10, in_channels=3)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=10, in_channels=3)


def Net_cifar100(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=100, in_channels=3)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=100, in_channels=3)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=100, in_channels=3)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=100, in_channels=3)
def Net_mnist(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=10, in_channels=1, cutout=False)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=10, in_channels=1, cutout=False)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=10, in_channels=1, cutout=False)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=10, in_channels=1, cutout=False)
def Net_purchase(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=2, in_channels=1, cutout=False)

def Net_adult(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=2, in_channels=1, cutout=False)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=2, in_channels=1, cutout=False)



# class All_CNN(nn.Module):
#     def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
#         super(All_CNN, self).__init__()
#         n_filter1 = int(96 * filters_percentage)
#         n_filter2 = int(192 * filters_percentage)
#         self.features = nn.Sequential(
#             Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
#             Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
#             Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
#             nn.Dropout(inplace=True) if dropout else Identity(),
#             Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
#             Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
#             Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
#             nn.Dropout(inplace=True) if dropout else Identity(),
#             Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
#             Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
#             nn.AvgPool2d(8),
#             Flatten(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(n_filter2, num_classes),
#         )
#
#     def forward(self, x):
#         features = self.features(x)
#         output = self.classifier(features)
#         return output
#
#
# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
#                  activation_fn=nn.ReLU, batch_norm=True, transpose=False):
#         if padding is None:
#             padding = (kernel_size - 1) // 2
#         model = []
#         if not transpose:
#             #             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#             #                                 )]
#             model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 bias=not batch_norm)]
#         else:
#             model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                          output_padding=output_padding, bias=not batch_norm)]
#         if batch_norm:
#             model += [nn.BatchNorm2d(out_channels, affine=True)]
#         model += [activation_fn()]
#         super(Conv, self).__init__(*model)
#
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
#
# class Flatten(nn.Module):
#     def __init__(self):
#         super(Flatten, self).__init__()
#
#     def forward(self, x):
#         return x.view(x.size(0), -1)