# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:09:20 2020

@author: user
"""
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


def model_init(data_name,device):
    if(data_name == 'mnist'):
        model = Net_mnist()
    elif(data_name == 'cifar10'):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net_cifar10(device=device)
    elif(data_name == 'purchase'):
        model = Net_purchase()
    elif(data_name == 'adult'):
        model = Net_adult()
    elif(data_name == 'simulate'):
        model=Net()
    return model

class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# def data_init(FL_params):
#     kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
#     trainset, testset = data_set(FL_params.data_name)
#     # shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
#     # trainset, shadow _trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)
#
#     # shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
#     # testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)
#     #
#
#     # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
#     # 构建测试数据加载器
#     test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
#     # shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
#
#     # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
#     split_index = [int(trainset.__len__() / FL_params.N_total_client)] * (FL_params.N_total_client - 1)
#     split_index.append(
#         int(trainset.__len__() - int(trainset.__len__() / FL_params.N_total_client) * (FL_params.N_total_client - 1)))
#     client_dataset = torch.utils.data.random_split(trainset, split_index)
#
#     # split_index = [int(shadow_trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
#     # split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
#     # shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
#     # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
#     client_loaders = []
#     # shadow_client_sloaders = []
#     for ii in range(FL_params.N_total_client):
#         client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
#         # shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
#         '''
#         By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
#         '''
#
#     return client_loaders, test_loader
class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net_purchase(nn.Module):
    def __init__(self):
        super(Net_purchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50,2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x   
    

class Net_adult(nn.Module):
    def __init__(self):
        super(Net_adult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10,2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x     



class Net_cifar10(nn.Module):
    def __init__(self, device='cpu'):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.flatten = nn.Flatten()
        # self.to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.flatten(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def data_set(FL_params,data_name):
    if not data_name in ['mnist', 'purchase', 'adult', 'cifar10', 'cifar100']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    if FL_params.data_name == 'mnist':
        trainset = datasets.MNIST('../data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))

        testset = datasets.MNIST('../data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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


class All_CNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(All_CNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )
      
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output  
    
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model) 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x  
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
    
    
    
    
    
    
    
    
    
        
    
