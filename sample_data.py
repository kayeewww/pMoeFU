#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import itertools
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

def get_subset(dataset, indices):
    # print(f"indices: {indices}, type: {type(indices)}")
    user_indices = np.array(indices)
    # print(f"indices: {user_indices}, type: {type(user_indices)}")
    return Subset(dataset, user_indices)

def create_user_dataloaders(dataset, dict_users, batch_size):
    # kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}
    user_dataloaders = []
    for user_id, indices in dict_users.items():
        subset = get_subset(dataset, indices)
        assert len(subset) > 0, "Subset dataset is empty or indices are not properly set."

        # 创建 DataLoader
        # data_loader = DataLoader(subset, batch_size=32, shuffle=True)

        # user_dataloaders[user_id] = DataLoader(subset, batch_size=batch_size)
        user_dataloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    return user_dataloaders

def agnews_noniid2(labels_train,labels_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    idxs = np.arange(len(labels_train),dtype=int)
    labels = np.array(labels_train)
    label_list = np.unique(labels)
    n_classes = len(label_list)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_test = np.arange(len(labels_test),dtype=int)
    labels_test = np.array(labels_test)
    label_list_test = np.unique(labels_test)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    #print(idxs_labels)
    idxs_test = idxs_labels_test[0,:]
    idxs_test = idxs_test.astype(int)

    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))

    for i in range(num_users):
    #Sample majority class for each user
        # print(i)
        unique_labels = np.unique(label_list)
        if len(unique_labels) >= 2:
            majority_labels = np.random.choice(unique_labels, 2, replace=False)
        else:
            # If there are fewer than 2 unique labels, use all available labels
            majority_labels = unique_labels
        # if(overlap):
        #     majority_labels = list(itertools.product(range(n_classes),repeat=2))
        #     majority_labels = random.choice(majority_labels)
        # else:
        #     unique_labels = np.unique(label_list)
        #
        #     # # Check if there are enough labels to sample from
        #     # if len(unique_labels) >= 2:
        #     #     majority_labels = np.random.choice(unique_labels, 2, replace=False)
        #     # else:
        #     #     majority_labels = unique_labels  # Use all available unique labels
        #
        #     majority_labels = np.random.choice(np.unique(label_list), 2, replace = False)
        #     label_list = np.array(list(set(label_list) - set(majority_labels)))

        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)

        #train set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        if len(majority_labels1_idxs) >= int(p * n_data / 2):
            sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p*n_data/2), replace = False)
        else:
            sub_data_idxs1 = []
        if len(majority_labels2_idxs) >= int(p * n_data / 2):
            sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p*n_data/2), replace = False)
        else:
            sub_data_idxs2 = []

        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1))
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs2))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))

        #validation set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        # sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p*n_data_val/2), replace = False)
        # sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p*n_data_val/2), replace = False)
        # Check if majority_labels2_idxs is empty before sampling
        if len(majority_labels1_idxs) >= int(p * n_data_val / 2):
            sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=False)
        else:
            sub_data_idxs1_val = []  # Handle empty array case appropriately
        if len(majority_labels2_idxs)>= int(p * n_data_val / 2):
            sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=False)
        else:
            sub_data_idxs2_val = []  # Handle empty array case appropriately

        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val))
        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs2_val))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))

        #test set
        majority_labels1_idxs_test = idxs_test[majority_labels[0] == labels_test[idxs_test]]
        majority_labels2_idxs_test = idxs_test[majority_labels[1] == labels_test[idxs_test]]
        if len(majority_labels1_idxs_test) >= int(p * n_data_test / 2):
            sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p*n_data_test/2), replace = False)
        else:
            sub_data_idxs1_test = []  # Handle empty array case appropriately
        # Check if majority_labels2_idxs_test is large enough
        if len(majority_labels2_idxs_test) >= int(p * n_data_test / 2):
            sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)
        else:
            # If there are fewer items than required, sample all available items
            sub_data_idxs2_test = majority_labels2_idxs_test

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test))
        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs2_test))

        #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test)))
        #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs2_test)))

    if p<1.0:
        for i in range(num_users):
            if(len(idxs)>=n_data):
                majority_labels = user_majority_labels[i]
                #train set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data), replace = False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                #validation set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data_val), replace = False)
                dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs11_val))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                #test set
                non_majority_labels1_idxs_test = idxs_test[(majority_labels[0] != labels_test[idxs_test]) & (majority_labels[1] != labels_test[idxs_test])]
                sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test, int((1-p)*n_data_test), replace = False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
                #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs11_test)))

            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    # for i in range(num_users):
    #     print("Train")
    #     majority_labels = user_majority_labels[i]
    #     print("client %d %.2f %d " %(i, (sum(labels[dict_users[i]] == majority_labels[0])+sum(labels[dict_users[i]] == majority_labels[0]))/len(dict_users[i]),len(dict_users[i]) ))
    #     print(majority_labels)
    #     if i == range(num_users)[-1]:
    #         print(10*"-")
    #
    # for i in range(num_users):
    #     print("Test")
    #     majority_labels = user_majority_labels[i]
    #     print("client %d %.2f %d " %(i, (sum(labels_test[dict_users_test[i]] == majority_labels[0])+sum(labels_test[dict_users_test[i]] == majority_labels[0]))/len(dict_users_test[i]), len(dict_users_test[i]) ))
    #     print(majority_labels)
    #     if i == range(num_users)[-1]:
    #         print(10*"-")

    return dict_users, dict_users_val, dict_users_test
def mnist_noniid2(dataset, dataset_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    """
    Create a non-IID partitioning of MNIST data for federated learning.

    Args:
        dataset (Dataset): The MNIST training dataset.
        dataset_test (Dataset): The MNIST testing dataset.
        num_users (int): The number of users (clients) to distribute the data to.
        p (float): The proportion of data for each user that should come from the majority classes.
        n_data (int): The number of training data samples per user.
        n_data_val (int): The number of validation data samples per user.
        n_data_test (int): The number of testing data samples per user.
        overlap (bool): Flag to determine if overlapping majority classes are allowed among users.

    Returns:
        dict: Training data indices for each user.
        dict: Validation data indices for each user.
        dict: Testing data indices for each user.
    """
    # Initialize indices and labels for training data
    idxs = np.arange(len(dataset), dtype=int)
    labels = dataset.train_labels.numpy()
    label_list = np.unique(labels)

    # Sort indices by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Initialize indices and labels for testing data
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = dataset_test.targets.numpy()
    label_list_test = np.unique(labels_test)

    # Sort indices by labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :].astype(int)

    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Define class labels and combinations
    num_classes = len(label_list)
    user_majority_labels = []

    for i in range(num_users):
        # Sample majority class for each user
        if overlap:
            majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        else:
            majority_labels = np.random.choice(label_list, 2, replace=False)

        user_majority_labels.append(majority_labels)

        # Assign training, validation, and testing data
        # Training set
        majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
        majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p * n_data / 2), replace=False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p * n_data / 2), replace=False)

        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1, sub_data_idxs2))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1) - set(sub_data_idxs2)))

        # Validation set
        majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
        majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=False)

        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val, sub_data_idxs2_val))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val) - set(sub_data_idxs2_val)))

        # Testing set
        majority_labels1_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[0]]
        majority_labels2_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[1]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p * n_data_test / 2), replace=False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test, sub_data_idxs2_test))

    # Distribute non-majority label data
    if p < 1.0:
        for i in range(num_users):
            if len(idxs) >= n_data:
                majority_labels = user_majority_labels[i]
                # Training set
                non_majority_label_idxs = idxs[(labels[idxs] != majority_labels[0]) & (labels[idxs] != majority_labels[1])]
                sub_data_idxs = np.random.choice(non_majority_label_idxs, int((1 - p) * n_data), replace=False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

                # Validation set
                non_majority_label_idxs = idxs[(labels[idxs] != majority_labels[0]) & (labels[idxs] != majority_labels[1])]
                sub_data_idxs_val = np.random.choice(non_majority_label_idxs, int((1 - p) * n_data_val), replace=False)
                dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs_val))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs_val)))

                # Testing set
                non_majority_label_idxs_test = idxs_test[(labels_test[idxs_test] != majority_labels[0]) & (labels_test[idxs_test] != majority_labels[1])]
                sub_data_idxs_test = np.random.choice(non_majority_label_idxs_test, int((1 - p) * n_data_test), replace=False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs_test))
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))
                # Debugging information to verify data distribution
    # for i in range(num_users):
    #     print(
    #         f"User {i}: mnist Training samples {len(dict_users[i])}, Validation samples {len(dict_users_val[i])}, Testing samples {len(dict_users_test[i])}")
    # print_user_data_distribution(dict_users, dataset, num_users)

    # target_class_to_remove = 0  # 例如，移除类别 0 的所有样本
    # updated_dict_users = remove_class_from_users(dict_users, dataset, target_class_to_remove)
    # print_user_data_distribution(updated_dict_users, dataset, num_users)

    return dict_users, dict_users_val, dict_users_test
def create_dataloaders_from_indices(dataset, updated_dict_users, unlearn_dict_users, batch_size=32, shuffle=True):
    """
    根据 updated_dict_users 和 unlearn_dict_users 创建新的数据集和 DataLoader。

    Args:
        dataset (Dataset): 原始数据集。
        updated_dict_users (dict): 包含用户索引的字典，用于生成更新后的数据集。
        unlearn_dict_users (dict): 包含被遗忘的数据索引的字典。
        batch_size (int, optional): 每个 DataLoader 的批次大小。默认为 32。
        shuffle (bool, optional): 是否对 DataLoader 中的数据进行随机打乱。默认为 True。

    Returns:
        dict: 每个用户的更新后的 DataLoader。
        dict: 每个用户的遗忘数据 DataLoader。
    """
    updated_dataloaders = {}
    unlearn_dataloaders = {}

    for user_id, indices in updated_dict_users.items():
        # 创建更新后的数据集
        updated_subset = Subset(dataset, indices)
        updated_dataloader = DataLoader(updated_subset, batch_size=batch_size, shuffle=shuffle)
        updated_dataloaders[user_id] = updated_dataloader
    for user_id, indices in unlearn_dict_users.items():
        # 确保 indices 是一个列表
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        elif isinstance(indices, np.ndarray):
            if indices.ndim == 0:
                indices = [indices.item()]
            else:
                indices = indices.tolist()
        unlearn_subset = torch.utils.data.Subset(dataset, indices)
        unlearn_dataloader = DataLoader(unlearn_subset, batch_size=1, shuffle=shuffle)
        unlearn_dataloaders[user_id] = unlearn_dataloader

    # for user_id, indices in unlearn_dict_users.items():
    #     # 创建遗忘数据集
    #     unlearn_subset = torch.utils.data.Subset(dataset, indices)
    #     unlearn_dataloader = DataLoader(unlearn_subset, batch_size=1, shuffle=shuffle)
    #     unlearn_dataloaders[user_id] = unlearn_dataloader

    return updated_dataloaders, unlearn_dataloaders
def create_user_removedloaders(dict_users, dataset, batch_size=64):
    """
    为每个用户生成 DataLoader。

    Args:
        dict_users (dict): 每个用户的数据集索引字典。
        dataset (Dataset): 原始数据集（例如 CIFAR-10）。
        batch_size (int): DataLoader 的批量大小。

    Returns:
        dict: 每个用户对应的 DataLoader 字典。
    """
    user_dataloaders = []


    for user_id, user_indices in dict_users.items():
        # print(f"indices: {user_indices}, type: {type(user_indices)}")
        user_indices = np.array(user_indices)
        # print(f"indices: {user_indices}, type: {type(user_indices)}")

        user_subset = Subset(dataset, user_indices)
        user_dataloader = DataLoader(user_subset, batch_size=batch_size, shuffle=True)
        # user_dataloaders[user_id] = user_dataloader
        user_dataloaders.append(user_dataloader)

    return user_dataloaders
def remove_class_from_users(dict_users, dataset, target_class_index):
    """
    从每个用户的数据集中移除指定类别的样本，并生成更新后的 dict_users。

    Args:
        dict_users (dict): 每个用户的数据集索引字典。
        dataset (Dataset): 原始数据集（例如 CIFAR-10）。
        target_class (int): 需要移除的类别标签。

    Returns:
        dict: 更新后的 dict_users。
    """
    updated_dict_users = {}
    unlearn_dict_users = {}
    # 获取数据集的标签
    labels = np.array(dataset.targets)

    # 遍历每个用户的数据集
    for user_id, user_data_indices in dict_users.items():
        # 获取该用户数据集的标签
        user_labels = labels[user_data_indices]

        # 找出不属于目标类别的样本索引
        remaining_indices = user_data_indices[user_labels != target_class_index]

        # 找出属于目标类别的样本索引
        unlearn_indices = user_data_indices[target_class_index]

        # 更新后的用户数据集索引
        updated_dict_users[user_id] = remaining_indices
        unlearn_dict_users[user_id] = unlearn_indices

    # print('updated_dict_users, unlearn_dict_users',updated_dict_users, unlearn_dict_users)
    # return dict_users, 0
    return updated_dict_users, unlearn_dict_users

def print_user_data_distribution(dict_users, dataset, num_users):
    """
    打印每个用户的数据集中的类别分布情况。

    Args:
        dict_users (dict): 每个用户的数据集索引字典。
        dataset (Dataset): 原始数据集（例如 CIFAR-10）。
        num_users (int): 用户数量。
    """
    for user_id in range(num_users):
        user_data_indices = dict_users[user_id]
        user_labels = np.array(dataset.targets)[user_data_indices]
        unique, counts = np.unique(user_labels, return_counts=True)
        print(f"用户 {user_id} 的数据集类别分布:")
        for label, count in zip(unique, counts):
            print(f"  类别 {label}： {count} 个样本")
        print("\n")
def cifar_noniid2(dataset, dataset_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    """
        Create a non-IID partitioning of CIFAR-10 data for federated learning.

        Args:
            dataset (Dataset): The CIFAR-10 training dataset.
            dataset_test (Dataset): The CIFAR-10 testing dataset.
            num_users (int): The number of users (clients) to distribute the data to.
            p (float): The proportion of data for each user that should come from the majority classes.
            n_data (int): The number of training data samples per user.
            n_data_val (int): The number of validation data samples per user.
            n_data_test (int): The number of testing data samples per user.
            overlap (bool): Flag to determine if overlapping majority classes are allowed among users.

        Returns:
            dict: Training data indices for each user.
            dict: Validation data indices for each user.
            dict: Testing data indices for each user.
        """
    # print('Here!!')
    # 数据索引 idxs 和标签数组 labels。
    idxs = np.arange(len(dataset), dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    # print(label_list)

    # # 对标签进行排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # print(idxs_labels)
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(dataset_test.targets)
    label_list_test = np.unique(dataset_test.targets)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    # print(idxs_labels)
    idxs_test = idxs_labels_test[0, :]
    idxs_test = idxs_test.astype(int)

    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 定义类标签和组合
    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))

    for i in range(num_users):
        # Sample majority class for each user
        majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        # overlap = True
        #
        # if (overlap):
        #     majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        # else:
        #     majority_labels = np.random.choice(np.unique(label_list), 2, replace=False)
        # print('majority_labels', majority_labels)
        label_list = np.array(list(set(label_list) - set(majority_labels)))
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)

        # 分配训练、验证和测试数据
        # train set
        # 训练集
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p * n_data / 2), replace=False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p * n_data / 2), replace=False)

        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1, sub_data_idxs2))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1) - set(sub_data_idxs2)))

        # dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1))
        # dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs2))
        # idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        # idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))

        # validation set
        # 验证集
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=False)

        # dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val))
        # dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs2_val))
        #
        # idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        # idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))

        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val, sub_data_idxs2_val))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val) - set(sub_data_idxs2_val)))


        # test set
        # 测试集
        majority_labels1_idxs_test = idxs_test[majority_labels[0] == labels_test[idxs_test]]
        majority_labels2_idxs_test = idxs_test[majority_labels[1] == labels_test[idxs_test]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p * n_data_test / 2), replace=False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test))
        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs2_test))
        # dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test, sub_data_idxs2_test))

        idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test)))
        idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs2_test)))

    # 分配非主要标签的数据
    if p < 1.0:
        for i in range(num_users):
            if (len(idxs) >= n_data):
                majority_labels = user_majority_labels[i]
                # train set
                non_majority_labels1_idxs = idxs[
                    (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data), replace=False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                # validation set
                non_majority_labels1_idxs = idxs[
                    (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1 - p) * n_data_val),
                                                       replace=False)
                dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs11_val))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))

                # test set
                non_majority_labels1_idxs_test = idxs_test[
                    (majority_labels[0] != labels_test[idxs_test]) & (majority_labels[1] != labels_test[idxs_test])]
                sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test, int((1 - p) * n_data_test),
                                                        replace=False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
                # idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs11_test)))

            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    # for i in range(num_users):
    #     print("Train")
    #     majority_labels = user_majority_labels[i]
    #     print("client %d %.2f %d " % (i, (sum(labels[dict_users[i]] == majority_labels[0]) + sum(
    #         labels[dict_users[i]] == majority_labels[0])) / len(dict_users[i]), len(dict_users[i])))
    #     # len(dict_users[i] =100
    #     print(majority_labels)
    #     if i == range(num_users)[-1]:
    #         print(10 * "-")
    #
    # for i in range(num_users):
    #     print("Test")
    #     majority_labels = user_majority_labels[i]
    #     print("client %d %.2f %d " % (i, (sum(labels_test[dict_users_test[i]] == majority_labels[0]) + sum(
    #         labels_test[dict_users_test[i]] == majority_labels[0])) / len(dict_users_test[i]), len(dict_users_test[i])))
    #     # len(dict_users_test[i]) =200
    #     print(majority_labels)  # [0 1-9]
    #     if i == range(num_users)[-1]:
    #         print(10 * "-")
    # print("Number of users: ", len(dict_users))
    # print(majority_labels)
    # print_user_data_distribution(dict_users, dataset, num_users)

    return dict_users, dict_users_val, dict_users_test


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
    # print_user_data_distribution(dict_users, dataset, num_users)
    return dict_users, dict_users_val, dict_users_test