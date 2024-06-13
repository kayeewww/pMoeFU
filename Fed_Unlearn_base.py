# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
#ourself libs
import total_variance as total_variance
from data_preprocess import data_set, model_init
from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain, test
from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity, select_least_important_clients


def load_model_pytorch(model, load_model, model_name):
    # print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    # 这段代码检查模型和加载的状态字典的键是否需要匹配
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # # just for vgg
    # # 这段代码处理特定情况下的VGG模型。它将加载的状态字典中的键中的'features.'替换为'features'，以及将'classifier.'替换为'classifier'
    # if model_name == "vgg":
    #     from collections import OrderedDict
    #
    #     load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
    #     load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            # print(key, model.state_dict()[key].shape)
        # print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            # print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, False)

def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):

    print(5*"#"+"  Federated Learning Start"+5*"#")
    std_time = time.time()
    if FL_params.pretrained==True:
        if FL_params.save_pretrained:
            print('-------Saving pre-trained model------')
            old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
            torch.save(old_GMs[-1].state_dict(), FL_params.pretrained_gms_file)
            torch.save(old_CMs[-1].state_dict(), FL_params.pretrained_cms_file)
            exit(0)
    elif FL_params.pretrained==False:
        old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)

    end_time = time.time()
    time_learn = (std_time - end_time)
    print(" Learning time consuming = {} secods".format(-time_learn))
    print(5*"#"+"  Federated Learning End"+5*"#")


    print('\n')
    """4.2 unlearning  a client，Federated Unlearning"""
    print(5*"#"+"  Federated Unlearning Start  "+5*"#")
    std_time = time.time()
    #Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
    FL_params.if_unlearning = True

    print('#' * 4, 'Client-level Unlearning', '#' * 4)
    if FL_params.pretrained:
        print('-------Using pre-trained model------')
        old_GMs = []
        old_CMs = []
        old_GM=model_init(FL_params.data_name,'cpu')
        old_CM=model_init(FL_params.data_name,'cpu')
        gm_checkpoint = torch.load(FL_params.pretrained_gms_file)
        old_GM.load_state_dict(gm_checkpoint, strict=False)
        # print(old_GM)
        cm_checkpoint = torch.load(FL_params.pretrained_cms_file)
        old_CM.load_state_dict(cm_checkpoint, strict=False)
        # incompatible_keys = old_GM.load_state_dict(gm_checkpoint, strict=False)

        for i in range(FL_params.global_epoch):
            old_GMs.append(old_GM)
            old_CMs.append(old_CM)

    # print('all',old_GMs)
    unlearn_GMs, train_acc, train_loss, test_acc, test_loss = total_variance.client_level_unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)

    end_time = time.time()
    time_unlearn = (std_time - end_time)

    print(" Client-level Unlearning time consuming = {} secods".format(-time_unlearn))
    print(5*"#"+"  Federated Unlearning End  "+5*"#")

    # uncali_unlearn_GMs=unlearn_GMs[:-1]
    return old_GMs, unlearn_GMs, old_CMs

    

def generate(dataset, list_classes: list):
    labels = []
    for label_name in list_classes:
        label_index = dataset.classes.index(label_name)  # 获取类别名称对应的索引
        labels.append(label_index)

    sub_dataset = []
    # for datapoint in dataset:
    #     _, label_index = datapoint  # Extract label
    #     if label_index in list_classes:
    #         sub_dataset.append(datapoint)
    # return sub_dataset
    for data, label_index in dataset:
        if label_index in labels:
            sub_dataset.append((data, label_index))

    return sub_dataset

