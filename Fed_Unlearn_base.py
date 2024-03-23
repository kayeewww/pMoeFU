# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
from pathlib import Path

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse

import torchvision
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time

from torchvision import transforms

from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity
# ourself libs
from model_initiation import model_init
from data_preprocess import data_set

from FL_base import fedavg, global_train_once, FL_Train, test


def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    """FL_train"""
    print(5 * "#" + "  Federated Learning Start" + 5 * "#")
    train_model, train_acc, train_epoch = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    print("train acc:%.4f" % train_acc)
    print("train epoch:%d" % train_epoch)
    print(5 * "#" + "  Federated Learning End" + 5 * "#")

    print('\n')
    """class pruner"""
    print(5 * "#" + "  Class Pruning Start  " + 5 * "#")
    #  Class_pruner(train_model, FL_params)
    print(5 * "#" + " Class Pruning End  " + 5 * "#")

    print('\n')
    """4.3 unlearning a client，Federated Unlearning without calibration"""
    print(5 * "#" + "  Federated Unlearning without Calibration Start  " + 5 * "#")

    print(5 * "#" + "  Federated Unlearning without Calibration End  " + 5 * "#")



def generate(dataset, list_classes: list):
    labels = []
    for label_id in list_classes:
        labels.append(list(dataset.classes)[int(label_id)])
    # print(labels)

    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)
    return sub_dataset

def Class_pruner(net, FL_params):
    if FL_params.data_name == 'cifar10':
        '''load data and model'''
        mean=[125.31 / 255, 122.95 / 255, 113.87 / 255]
        std=[63.0 / 255, 62.09 / 255, 66.70 / 255]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        project_dir = Path(__file__).resolve().parent
        trainset = torchvision.datasets.CIFAR10(root=project_dir / 'data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=project_dir / 'data', train=False, download=True, transform=transform_test)
        # net = load_model_CIFAR10(args, model_path)
        total_classes = 10 # [0-9]

    train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)

    '''pre-processing'''
    feature_iit, classes = acculumate_feature(net, train_all_loader, 1)
    tf_idf_map = calculate_cp(feature_iit, classes, FL_params.data_name, 0, unlearn_class=FL_params.unlearn_class)
    threshold = get_threshold_by_sparsity(tf_idf_map, FL_params.sparsity)
    print('threshold', threshold)

    '''test before pruning'''
    list_allclasses = list(range(total_classes))
    unlearn_listclass = [FL_params.unlearn_class]
    list_allclasses.remove(FL_params.unlearn_class)  # rest classes
    unlearn_testdata = generate(testset, unlearn_listclass)
    rest_testdata = generate(testset, list_allclasses)
    print(len(unlearn_testdata), len(rest_testdata))
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=FL_params.test_batch_size,
                                                     shuffle=False, num_workers=4)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=FL_params.test_batch_size,
                                                  shuffle=False, num_workers=4)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)

    device_cpu = torch.device("cpu")
    net.to(device_cpu)
    test(net, unlearn_testloader)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(net, rest_testloader)
    print('*' * 40)