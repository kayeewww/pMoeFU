# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:39:07 2020

@author: user
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,TensorDataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from scipy.sparse import load_npz
import torchtext
import itertools

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

from sample_data import mnist_noniid2, cifar_noniid2, agnews_noniid2
from Models import (CNNCifar, CNNFashion, GateCNN, GateResNet, GateCNNFashion, LSTMGate, LSTMClassifier, GateModel,
                    GateNetPurchase,GateNetAdult)
from LanguageModels import RNNGate, RNNTextClassifier


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def get_dataloaders(train_data, val_data, block_size, batch_size):
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


# Hyperparameter tuning example
# def hyperparameter_tuning(model, train_loader):
#     param_grid = {
#         'lr': [0.001, 0.01, 0.1],
#         'batch_size': [32, 64, 128]
#     }
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
#     grid_search.fit(train_loader.dataset, train_loader.targets)
#
#     print("Best parameters found: ", grid_search.best_params_)

def data_set(data_name):
    # if not data_name in ['mnist', 'purchase', 'adult', 'cifar10']:
    #     raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')

    # model: 2 conv. layers followed by 2 FC layers
    trainset, testset=0,0
    if (data_name == 'cnn_mnist'):
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

    # model: ResNet-50
    elif (data_name == 'cnn_cifar10' or data_name == 'res_cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)

        testset = datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)

    # model: 2 FC layers
    elif (data_name == 'fcn_purchase'):
        data = np.concatenate([load_npz("./data/purchase/data1.npz").toarray(),
                               load_npz("./data/purchase/data2.npz").toarray()]).astype(int)
        num_class = 2
        if not os.path.exists(f"./data/purchase/{num_class}_kmeans.npy"):
            kmeans = KMeans(n_clusters=num_class, random_state=0).fit(data)
            label = kmeans.labels_
            np.save(f"./data/purchase/{num_class}_kmeans.npy", label)
        else:
            label = np.load(f"./data/purchase/{num_class}_kmeans.npy")
        # xx = np.load("./data/purchase/purchase_xx.npy")
        # yy = np.load("./data/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)


    # model: 2 FC layers
    elif (data_name == 'fcn_adult'):
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

        # trainset = Array2Dataset(xx_train, yy_train)
        # testset = Array2Dataset(xx_test, yy_test)
        trainset = TensorDataset(xx_train, yy_train)
        testset = TensorDataset(xx_test, yy_test)
    elif(data_name=='fashion'):
        trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                               transform=trans_fashionmnist)
        testset = datasets.FashionMNIST('../data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashionmnist)

    return trainset, testset

def noniid_partition(dataset, dataset_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    idxs = np.arange(len(dataset), dtype=int)
    labels = np.array([y for _, y in dataset])
    label_list = np.unique(labels)
    # 使用更多标签组合
    combinations = list(itertools.product(range(len(label_list)), repeat=5))
    while len(combinations) < num_users:
        combinations.extend(combinations)
    combinations = combinations[:num_users]

    if num_users > len(list(itertools.product(range(len(label_list)), repeat=5))):
        raise ValueError("Number of users exceeds the number of unique label combinations")

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array([y for _, y in dataset_test])
    label_list_test = np.unique(labels_test)

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :].astype(int)

    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))


    for i in range(num_users):
        majority_labels = combinations[i]
        # if overlap:
        #     majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        # else:
        #     majority_labels = np.random.choice(label_list, 2, replace=False)

        # majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        label_list = np.array(list(set(label_list) - set(majority_labels)))
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)

        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p * n_data / 2), replace=False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p * n_data / 2), replace=False)

        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1, sub_data_idxs2))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1) - set(sub_data_idxs2)))

        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=False)

        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val, sub_data_idxs2_val))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val) - set(sub_data_idxs2_val)))

        majority_labels1_idxs_test = idxs_test[majority_labels[0] == labels_test[idxs_test]]
        majority_labels2_idxs_test = idxs_test[majority_labels[1] == labels_test[idxs_test]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p * n_data_test / 2), replace=False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test, sub_data_idxs2_test))
        idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test) - set(sub_data_idxs2_test)))

    # if p < 1.0:
    #     for i in range(num_users):
    #         if len(idxs) >= n_data:
    #             majority_labels = user_majority_labels[i]
    #             non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
    #             sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data), replace=False)
    #             dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
    #             idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
    #
    #             non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
    #             sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data_val), replace=False)
    #             dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs11_val))
    #             idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
    #
    #             non_majority_labels1_idxs_test = idxs_test[(majority_labels[0] != labels_test[idxs_test]) & (majority_labels[1] != labels_test[idxs_test])]
    #             sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test, int((1 - p) * n_data_test), replace=False)
    #             dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
    #         else:
    #             dict_users[i] = np.concatenate((dict_users[i], idxs))
    #             dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))
    if p < 1.0:
        for i in range(num_users):
            if len(idxs) >= n_data:
                majority_labels = user_majority_labels[i]
                non_majority_labels1_idxs = idxs[
                    (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                if len(non_majority_labels1_idxs) >= int((1 - p) * n_data):
                    sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data), replace=False)
                    dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
                    idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                    non_majority_labels1_idxs = idxs[
                        (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                    if len(non_majority_labels1_idxs) >= int((1 - p) * n_data_val):
                        sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data_val),
                                                               replace=False)
                        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs11_val))
                        idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                        non_majority_labels1_idxs_test = idxs_test[
                            (majority_labels[0] != labels_test[idxs_test]) & (
                                        majority_labels[1] != labels_test[idxs_test])]
                        if len(non_majority_labels1_idxs_test) >= int((1 - p) * n_data_test):
                            sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test,
                                                                    int((1 - p) * n_data_test),
                                                                    replace=False)
                            dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
                else:
                    dict_users[i] = np.concatenate((dict_users[i], idxs))
                    dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    return dict_users, dict_users_val, dict_users_test


def load_and_partition_data(dataset_name,user_count, FL_params):
    dataset_train, dataset_test = data_set(dataset_name)

    if dataset_name == 'res_cifar10' or dataset_name == 'cnn_cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        dict_users, dict_users_val, dict_users_test = cifar_noniid2(
            dataset_train, dataset_test, user_count, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'cnn_mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform)
        dict_users, dict_users_val, dict_users_test = mnist_noniid2(
            dataset_train, dataset_test, user_count, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'cnn_fashion':
        dataset_train, dataset_test = data_set('fashion')
        dict_users, dict_users_val, dict_users_test = cifar_noniid2(
            dataset_train, dataset_test, user_count, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'fcn_purchase':
        dict_users, dict_users_val, dict_users_test = noniid_partition(
            dataset_train, dataset_test, user_count, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'fcn_adult':
        dict_users, dict_users_val, dict_users_test = noniid_partition(
            dataset_train, dataset_test, user_count, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'agnews':
        TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
        LABEL = torchtext.data.LabelField(is_target=True)
        datafields = [('text', TEXT), ('label', LABEL)]
        dataset_train = read_data(
            './data/agnews/train.csv',
            datafields, label_column=0, doc_start=2)
        dataset_test = read_data(
            './data/agnews/test.csv',
            datafields, label_column=0, doc_start=2)
        TEXT.build_vocab(dataset_train, max_size=10000)
        LABEL.build_vocab(dataset_test)

        labels = [LABEL.vocab.stoi[l] for l in dataset_train.label]
        labels_test = [LABEL.vocab.stoi[l.split(',')[0]] for l in dataset_test.label]
        dict_users, dict_users_val, dict_users_test = agnews_noniid2(labels, labels_test, user_count, FL_params.p,
                                                                     FL_params.n_data, FL_params.n_data_val,
                                                                     FL_params.n_data_test, FL_params.overlap)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test
def read_data(corpus_file, datafields, label_column, doc_start):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        for line in f:
            columns = line.strip().split(maxsplit=doc_start)
            doc = columns[-1]
            # Split the label and take the first part if needed
            label = columns[label_column].split(',')[0]  # Assuming the label is in format '4,Prediction'
            examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
    return torchtext.data.Dataset(examples, datafields)
def splitExpertData(FL_params):
    dataset_train = []
    dataset_test = []
    dict_users = []
    dict_users_val = []
    dict_users_test = []

    # 加载第一个数据集
    data_1_train, data_1_test, data_1_users, data_1_users_val, data_1_users_test = load_and_partition_data(FL_params.data_1, FL_params)
    dataset_train.append(data_1_train)
    dataset_test.append(data_1_test)
    dict_users.append(data_1_users)
    dict_users_val.append(data_1_users_val)
    dict_users_test.append(data_1_users_test)

    # 加载第二个数据集，如果和第一个数据集不同
    if FL_params.model_1 != FL_params.model_2:
        data_2_train, data_2_test, data_2_users, data_2_users_val, data_2_users_test = load_and_partition_data(FL_params.data_2, FL_params)
        dataset_train.append(data_2_train)
        dataset_test.append(data_2_test)
        dict_users.append(data_2_users)
        dict_users_val.append(data_2_users_val)
        dict_users_test.append(data_2_users_test)
    else:
        # 如果两个数据集相同，则复制第一个数据集的信息
        dataset_train.append(data_1_train)
        dataset_test.append(data_1_test)
        dict_users.append(data_1_users)
        dict_users_val.append(data_1_users_val)
        dict_users_test.append(data_1_users_test)

    return dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test


# def splittExpertModel(FL_params):
#     tfidf_dim=20
#
#     net_glob_fedAvg, gate_model, net_locals, client_models=[],[],[], {}
#     if FL_params.model_1 == 'resnet_cifar' or FL_params.model_2 == 'resnet_cifar':
#         input_shape_cifar = (3, 32, 32)  # CIFAR-10 输入形状
#         conv_layers_cifar = [(6, 5), (16, 5)]  # CIFAR-10 卷积层配置
#         linear_layers_cifar = [120, 84]  # CIFAR-10 全连接层配置
#
#         net_glob_fedAvg_cifar = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
#         gate_model_cifar = GateModel(input_shape_cifar, conv_layers_cifar, linear_layers_cifar, tfidf_dim, FL_params.type_count_2nd)
#         if FL_params.seperate_mix:
#             gate_model_cifar = GateResNet(args=FL_params).to(FL_params.device)
#         net_locals_cifar = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
#         client_models.update({
#             'local_2': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#             'global_2': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_2': gate_model_cifar  # gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_1': net_glob_fedAvg_cifar,  # net_locals_adult,#net_locals2,
#                 'global_1': net_locals_cifar,  # net_glob_fedAvg2,
#                 'gate_1': gate_model_cifar,  # gate_model2
#             })
#
#     if FL_params.model_1 == 'cnn_cifar' or FL_params.model_2 == 'cnn_cifar':
#         input_shape_cifar = (3, 32, 32)  # CIFAR-10 输入形状
#         conv_layers_cifar = [(6, 5), (16, 5)]  # CIFAR-10 卷积层配置
#         linear_layers_cifar = [120, 84]  # CIFAR-10 全连接层配置
#
#         net_glob_fedAvg_cifar = CNNCifar(args=FL_params).to(FL_params.device)
#         gate_model_cifar = GateModel(input_shape_cifar, conv_layers_cifar, linear_layers_cifar, tfidf_dim, FL_params.type_count_2nd)
#         if FL_params.seperate_mix:
#             gate_model_cifar = GateResNet(args=FL_params).to(FL_params.device)
#         net_locals_cifar =  CNNCifar(args=FL_params).to(FL_params.device)
#         client_models.update({
#             'local_1': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#             'global_1': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_1': gate_model_cifar  # gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#                 'global_2': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#                 'gate_2': gate_model_cifar  # gate_model_purchase#
#             })
#     if FL_params.model_1 == 'cnn_mnist' or FL_params.model_2 == 'cnn_mnist' or FL_params.model_1 == 'cnn_fashion'or FL_params.model_2 == 'cnn_fashion':
#         input_shape_mnist = (1, 28, 28)  # MNIST 输入形状
#         conv_layers_mnist = [(6, 5), (12, 5)]  # MNIST 卷积层配置
#         linear_layers_mnist = [84, 42]  # MNIST 全连接层配置
#         net_glob_fedAvg_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         gate_model_mnist = GateModel(input_shape_mnist, conv_layers_mnist, linear_layers_mnist,tfidf_dim, FL_params.type_count_1st)
#         if FL_params.seperate_mix:
#             gate_model_mnist = GateCNNFashion(args=FL_params).to(FL_params.device)
#         net_locals_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         client_models.update({
#             'local_2': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#             'global_2': net_locals_mnist,  # net_glob_fedAvg2,
#             'gate_2': gate_model_mnist,  # gate_model2
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_1': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#                 'global_1': net_locals_mnist,  # net_glob_fedAvg2,
#                 'gate_1': gate_model_mnist,  # gate_model2
#             })
#
#     if FL_params.model_1 == 'purchase' or FL_params.model_2 == 'purchase':
#         input_shape_purchase = (600,)  # Purchase 数据集输入形状
#         linear_layers_purchase = [300, 50]  # Purchase 数据集全连接层配置
#         # tfidf_dim_purchase = 20
#         num_experts_purchase = 5
#         net_glob_fedAvg_purchase = Net_purchase().to(FL_params.device)
#         gate_model_purchase = GateModel(input_shape_purchase, 0, linear_layers_purchase, tfidf_dim, num_experts_purchase).to(
#             FL_params.device)
#
#         # gate_model_purchase = GateModel(input_shape_purchase,0, linear_layers_purchase, tfidf_dim=5, num_experts=FL_params.type_count_2nd)#purchase_count)
#         if FL_params.seperate_mix:
#             gate_model_purchase = GateNetPurchase().to(FL_params.device)
#         net_locals_purchase = Net_purchase().to(FL_params.device)
#         client_models.update({
#             'local_1':  net_locals_purchase,#net_locals1,
#             'global_1':  net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_1':  gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_locals_purchase,  # net_locals1,
#                 'global_2': net_glob_fedAvg_purchase,  # net_glob_fedAvg1,
#                 'gate_2': gate_model_purchase  #
#             })
#
#     if FL_params.model_1 == 'adult' or FL_params.model_2 == 'adult':
#         input_shape_adult = (108,)  # Adult 数据集输入形状
#         linear_layers_adult = [50, 10]  # Adult 数据集全连接层配置
#         # tfidf_dim_purchase = 20
#         num_experts_purchase = 5
#
#         net_glob_fedAvg_adult = Net_adult().to(FL_params.device)
#         gate_model_adult = GateModel(input_shape_adult,0, linear_layers_adult, tfidf_dim,
#                                         num_experts_purchase).to(
#             FL_params.device)
#
#         if FL_params.seperate_mix:
#             gate_model_adult = GateNetAdult().to(FL_params.device)
#         net_locals_adult = Net_adult().to(FL_params.device)
#         client_models.update({
#             'local_2':  net_locals_adult,#net_locals2,
#             'global_2':  net_glob_fedAvg_adult,
#             'gate_2':  gate_model_adult
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_1': net_locals_adult,  # net_locals2,
#                 'global_1': net_glob_fedAvg_adult,
#                 'gate_1': gate_model_adult
#             })
#
#     return (client_models['global_1'], client_models['gate_1'],client_models['local_1'],
#             client_models['global_2'], client_models['gate_2'], client_models['local_2'],client_models)
# def splittExpertModel(FL_params):
#
#     net_glob_fedAvg, gate_model, net_locals, client_models=[],[],[], {}
#     if FL_params.model_1 == 'resnet_cifar' or FL_params.model_2 == 'resnet_cifar':
#         net_glob_fedAvg_cifar = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
#         gate_model_cifar = GateResNet(args=FL_params).to(FL_params.device)
#         net_locals_cifar = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
#         client_models.update({
#             'local_1': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#             'global_1': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_1': gate_model_cifar  # gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_glob_fedAvg_cifar,  # net_locals_adult,#net_locals2,
#                 'global_2': net_locals_cifar,  # net_glob_fedAvg2,
#                 'gate_2': gate_model_cifar,  # gate_model2
#             })
#
#     if FL_params.model_1 == 'cnn_cifar' or FL_params.model_2 == 'cnn_cifar':
#         net_glob_fedAvg_cifar = CNNCifar(args=FL_params).to(FL_params.device)
#         gate_model_cifar = GateCNN(args=FL_params).to(FL_params.device)
#         net_locals_cifar = CNNCifar(args=FL_params).to(FL_params.device)
#         client_models.update({
#             'local_1': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#             'global_1': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_1': gate_model_cifar  # gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_glob_fedAvg_cifar,  # net_locals_purchase,#net_locals1,
#                 'global_2': net_locals_cifar,  # net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#                 'gate_2': gate_model_cifar  # gate_model_purchase#
#             })
#     if FL_params.model_1 == 'cnn_mnist' or FL_params.model_2 == 'cnn_mnist':
#         net_glob_fedAvg_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         # gate_model_mnist = GateModel(input_shape_mnist, conv_layers_mnist, linear_layers_mnist,tfidf_dim, FL_params.type_count_1st)
#         gate_model_mnist = GateCNNFashion(args=FL_params).to(FL_params.device)
#         net_locals_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         client_models.update({
#             'local_2': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#             'global_2': net_locals_mnist,  # net_glob_fedAvg2,
#             'gate_2': gate_model_mnist,  # gate_model2
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_1': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#                 'global_1': net_locals_mnist,  # net_glob_fedAvg2,
#                 'gate_1': gate_model_mnist,  # gate_model2
#             })
#     if FL_params.model_1 == 'cnn_fashion'or FL_params.model_2 == 'cnn_fashion':
#         net_glob_fedAvg_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         # gate_model_mnist = GateModel(input_shape_mnist, conv_layers_mnist, linear_layers_mnist,tfidf_dim, FL_params.type_count_1st)
#         gate_model_mnist = GateCNNFashion(args=FL_params).to(FL_params.device)
#         net_locals_mnist = CNNFashion(args=FL_params).to(FL_params.device)
#         client_models.update({
#             'local_1': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#             'global_1': net_locals_mnist,  # net_glob_fedAvg2,
#             'gate_1': gate_model_mnist,  # gate_model2
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_glob_fedAvg_mnist,  # net_locals_adult,#net_locals2,
#                 'global_2': net_locals_mnist,  # net_glob_fedAvg2,
#                 'gate_2': gate_model_mnist,  # gate_model2
#             })
#     if FL_params.model_1 == 'purchase' or FL_params.model_2 == 'purchase':
#         net_glob_fedAvg_purchase = Net_purchase().to(FL_params.device)
#         gate_model_purchase = GateNetPurchase().to(FL_params.device)
#         net_locals_purchase = Net_purchase().to(FL_params.device)
#         client_models.update({
#             'local_1':  net_locals_purchase,#net_locals1,
#             'global_1':  net_glob_fedAvg_purchase,#net_glob_fedAvg1,
#             'gate_1':  gate_model_purchase#
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_2': net_locals_purchase,  # net_locals1,
#                 'global_2': net_glob_fedAvg_purchase,  # net_glob_fedAvg1,
#                 'gate_2': gate_model_purchase  #
#             })
#
#     if FL_params.model_1 == 'adult' or FL_params.model_2 == 'adult':
#         net_glob_fedAvg_adult = Net_adult().to(FL_params.device)
#         gate_model_adult = GateNetAdult().to(FL_params.device)
#         net_locals_adult = Net_adult().to(FL_params.device)
#         client_models.update({
#             'local_2':  net_locals_adult,#net_locals2,
#             'global_2':  net_glob_fedAvg_adult,
#             'gate_2':  gate_model_adult
#         })
#         if FL_params.model_1 == FL_params.model_2:
#             client_models.update({
#                 'local_1': net_locals_adult,  # net_locals2,
#                 'global_1': net_glob_fedAvg_adult,
#                 'gate_1': gate_model_adult
#             })
#
#     return (client_models['global_1'], client_models['gate_1'],client_models['local_1'],
#             client_models['global_2'], client_models['gate_2'], client_models['local_2'],client_models)
def create_models(model_name, FL_params):
    if model_name == 'res_cifar10':
        net_glob = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
        gate_model = GateResNet(args=FL_params).to(FL_params.device)
        net_local = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
    elif model_name == 'cnn_cifar10':
        net_glob = CNNCifar(args=FL_params).to(FL_params.device)
        gate_model = GateCNN(args=FL_params).to(FL_params.device)
        net_local = CNNCifar(args=FL_params).to(FL_params.device)
    elif model_name == 'cnn_mnist':
        net_glob = CNNFashion(args=FL_params).to(FL_params.device)
        gate_model = GateCNNFashion(args=FL_params).to(FL_params.device)
        net_local = CNNFashion(args=FL_params).to(FL_params.device)
    elif model_name == 'cnn_fashion':
        net_glob = CNNFashion(args=FL_params).to(FL_params.device)
        gate_model = GateCNNFashion(args=FL_params).to(FL_params.device)
        net_local = CNNFashion(args=FL_params).to(FL_params.device)
    elif model_name == 'fcn_purchase':
        net_glob = Net_purchase().to(FL_params.device)
        gate_model = GateNetPurchase().to(FL_params.device)
        net_local = Net_purchase().to(FL_params.device)
    elif model_name == 'fcn_adult':
        net_glob = Net_adult().to(FL_params.device)
        gate_model = GateNetAdult().to(FL_params.device)
        net_local = Net_adult().to(FL_params.device)
    elif model_name == 'agnews':
        TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
        LABEL = torchtext.data.LabelField(is_target=True)
        datafields = [('text', TEXT), ('label', LABEL)]
        # ds = load_dataset("fancyzhx/ag_news")
        # train=ds['train']
        # test=ds['test']

        train = read_data(
            '/Users/wangjiayi/Downloads/federated-learning/data/agnews/train.csv',
            datafields, label_column=0, doc_start=2)
        test = read_data(
            '/Users/wangjiayi/Downloads/federated-learning/data/agnews/test.csv',
            datafields, label_column=0, doc_start=2)

        # train = read_data('/home/edvinli/data/agnews/ag_news.train', datafields, label_column=0, doc_start=2)
        # test = read_data('/home/edvinli/data/agnews/ag_news.test', datafields, label_column=0, doc_start=2)
        TEXT.build_vocab(train, max_size=10000)
        LABEL.build_vocab(train)
        labels = [LABEL.vocab.stoi[l] for l in train.label]
        labels_test = [LABEL.vocab.stoi[l] for l in test.label]

        net_glob = RNNTextClassifier(TEXT, LABEL, emb_dim=100, rnn_size=64).to(FL_params.device)
        gate_model = RNNGate(TEXT, LABEL, emb_dim=100, rnn_size=64).to(FL_params.device)
        net_local = RNNTextClassifier(TEXT, LABEL, emb_dim=100, rnn_size=64).to(FL_params.device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return net_glob, gate_model, net_local


def splittExpertModel(FL_params):
    client_models = {}

    net_glob_1, gate_model_1, net_local_1 = create_models(FL_params.model_1, FL_params)
    client_models.update({
        'local_1': net_local_1,
        'global_1': net_glob_1,
        'gate_1': gate_model_1
    })

    if FL_params.model_1 == FL_params.model_2:
        client_models.update({
            'local_2': net_local_1,
            'global_2': net_glob_1,
            'gate_2': gate_model_1
        })
    else:
        net_glob_2, gate_model_2, net_local_2 = create_models(FL_params.model_2, FL_params)
        client_models.update({
            'local_2': net_local_2,
            'global_2': net_glob_2,
            'gate_2': gate_model_2
        })

    return client_models


"""Function: load data"""


def dataloader_init(data_name, FL_params):
    kwargs = {'num_workers': 1, 'pin_memory': True} if FL_params.cuda_state else {}

    trainset, testset = data_set(data_name)
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
    split_index = [int(trainset.__len__() / FL_params.N_client)] * (FL_params.N_client - 1)
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_params.N_client) * (FL_params.N_client - 1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
    client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
        '''
        By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
        '''

    return trainset, testset, client_loaders, test_loader


# define class->dataset  for adult and purchase datasets
# for the purchase, we use TensorDataset function to transform numpy.array to datasets class
# for the adult, we custom an AdultDataset class that inherits torch.util.data.Dataset class
# """
# Array2Dataset: A class that can transform np.array(tensor matrix) to a torch.Dataset class.
# """
# class Array2Dataset(Dataset):
#     def __init__(self, data, targets, transform=None):
#         self.data = data
#         self.targets = targets
#         self.transform = transform
#     def __getitem__(self, index):
#         x = self.data[index,:]
#         y = self.targets[index]
#         return x, y
#     def __len__(self):
#         return len(self.data)

###################################MODEL##########################################
def model_init(data_name, device):
    if (type(data_name) == list):
        model_list = []
        for i in data_name:
            if ((i == 'mix_cifar10')):
                model_list.append(Net_cifar10_new())
            elif ((i == 'mix_mnist')):
                model_list.append(Net_mnist_new())
        return model_list
    else:
        if (data_name == 'mnist'):
            model = Net_mnist()
        elif (data_name == 'cifar10'):
            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            model = Net_cifar10(device)
        elif (data_name == 'purchase'):
            model = Net_purchase()
        elif (data_name == 'adult'):
            model = Net_adult()
        elif (data_name == 'simulate'):
            model = Net()
            # elif (data_name == 'shakespeare'):
            #     text, data, string2integer, integer2string, vocab_size, chars = load_data("data/shakespeare.txt")
            #     config = GPTConfig(
            #         block_size=4,
            #         vocab_size=len(chars),
            #         n_head=4,
            #         n_layer=4,
            #         n_embd=16
            #     )
            #
            #     model = BabyGPTmodel(config)
            #     model.to(device)

            return model


class Net(nn.Module):
    def __init__(self, device, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_cifar10(nn.Module):

    def __init__(self, device):
        super(Net_cifar10, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.flatten = nn.Flatten()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Expert1 input shape: {x.shape}")
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.flatten(x)
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_mnist(nn.Module):
    def __init__(self, num_classes=10):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.input_size = (1, 28, 28)
        # self.output_size = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class Net_cifar10_new(nn.Module):

    def __init__(self):
        super(Net_cifar10_new, self).__init__()
        # self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64 * 32 * 32)  # , 10)
        # self.flatten = nn.Flatten()
        # self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Expert1 input shape: {x.shape}")
        # x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.flatten(x)
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 16 * 5 * 5)
        # print(f"Expert1 reshaped input shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 64, 32, 32)


class Net_mnist_new(nn.Module):
    # def __init__(self, num_classes=10):
    #     super(Net_mnist_new, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    #     self.fc1 = nn.Linear(64 * 32 * 32, 128)
    #     # self.fc2 = nn.Linear(128)
    #     self.fc2 = nn.Linear(128, 64 * 32 * 32)# num_classes)
    #     # self.input_size = (1, 28, 28)
    #     # self.output_size = num_classes
    #
    # def forward(self, x):
    #     # print(f"Expert2 input shape: {x.shape}")
    #     x = x[:, 0, :, :].unsqueeze(1)
    #     # print(f"Expert2 input One channel: {x.shape}")
    #
    #     x = self.conv1(x)
    #     x = torch.relu(x)
    #     x = self.conv2(x)
    #     x = torch.relu(x)
    #     x = x.view(x.size(0), -1)
    #     # print(f"Expert2 reshaped input shape: {x.shape}")  # 64*32*32
    #     x = self.fc1(x)
    #     x = torch.relu(x)
    #     x = self.fc2(x)
    #     # x.view(-1, 64, 32, 32)
    #     # x = x.view(x.size(0), 1, 10, 10)  # Adjusting the output to have single channel
    #     x = x.repeat(16, 64, 1, 1)  # Repeating channels to make it 3-channel output
    #     x = F.interpolate(x, size=(32, 32))  # Resize to (3, 32, 32)
    #     return x.view(-1, 64, 32, 32)
    def __init__(self):
        super(Net_mnist_new, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Adjusted to 3 input channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 120)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64 * 32 * 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Adjusted input size after pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 64, 32, 32)


class Net_purchase(nn.Module):
    def __init__(self):
        super(Net_purchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 2)

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
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


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

    def forward(self, x):
        return x.view(x.size(0), -1)
