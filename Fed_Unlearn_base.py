# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
from pathlib import Path
import re
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
from data_preprocess import data_set, data_setted

from FL_base import fedavg, global_train_once, FL_Train, test
# from Fed_pruner import generate
from torch.utils.data import Dataset, DataLoader
# from Fed_Unlearn_main import Arguments


class CustomDataset(Dataset):
    def __init__(self, data, num_samples):
        self.data = data
        self.num_samples = num_samples #if num_samples is not None else len(data)

    def __len__(self):
        return min(len(self.data), self.num_samples)

    def __getitem__(self, idx):
        # 返回数据集中指定索引处的数据样本和对应的标签
        sample, label = self.data[idx]  # 假设数据样本是一个元组，包含数据和标签
        return sample, label
class MyDataset(Dataset):
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回数据和对应的类别索引
        return self.data[idx], self.classes[idx]

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
def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    """FL_train"""
    print(5 * "#" + " Federated Learning Start " + 5 * "#")
    std_time = time.time()
    train_model, val_acc, old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = end_time-std_time
    print("Time for FL: ", time_learn, 's')
    # print("Federated Learning train acc:%.4f" % val_acc)
    # print("Federated Learning train epoch:%d" % FL_params.global_epoch) #train_epoch)
    print(5 * "#" + "  Federated Learning End " + 5 * "#")

    print('\n')
    """class pruner"""
    print(5 * "#" + "  Class Pruning Start  " + 5 * "#")
    std_time = time.time()
    FL_params.if_unlearning = True

    # TODO 具体的剪枝过程
    mode='client'
    FL_params.unlearn_class = Class_pruner(train_model, FL_params)

    print(5 * "#" + " Class Pruning End  " + 5 * "#")

    unlearn_GMs = unlearning(mode, old_GMs, old_CMs, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_unlearn = end_time - std_time
    print("Time for UL: ", time_unlearn, 's')
    print('\n')

    """4.3 unlearning a client，Federated Unlearning without calibration"""
    print(5 * "#" + "  Federated Unlearning without Calibration Start  " + 5 * "#")
    std_time = time.time()
    uncali_unlearn_GMs = unlearning_without_cali(old_GMs, old_CMs, FL_params)
    end_time = time.time()
    time_unlearn_no_cali = end_time-std_time
    print(5 * "#" + "  Federated Unlearning without Calibration End  " + 5 * "#")

    print(" Learning time consuming = {} secods".format(round(time_learn, 3)))
    print(" Unlearning time consuming = {} secods".format(round(time_unlearn, 3)))
    print(" Unlearning no Cali time consuming = {} secods".format(round(time_unlearn_no_cali, 3)))
    # print(" Retraining time consuming = {} secods".format(-time_retrain))

    return old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs

def Class_pruner(net, FL_params):

    '''load data and model'''
    if FL_params.data_name == 'cifar10':
        net = model_init('cifar10',FL_params.model_name)
        trainset, testset = data_set('cifar10')
        total_classes = 10 # [0-9]
    elif FL_params.data_name == 'mnist':
        trainset,testset=data_set('mnist')
        net = model_init('mnist', FL_params.model_name)
        total_classes = 10  # [0-9]
    elif FL_params.data_name == 'cifar100':
        trainset,testset=data_set('cifar100')
        net = model_init('cifar100', FL_params.model_name)
        total_classes = 10  # [0-9]
    elif FL_params.data_name == 'adult':
        trainset,testset=data_set('adult')
        net = model_init('adult', FL_params.model_name)
        total_classes = 2  # [0-1]
    elif FL_params.data_name == 'purchase':
        trainset,testset=data_set('purchase')
        net = model_init('purchase', FL_params.model_name)
        total_classes = 2  # [0-9]
    train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)

    '''pre-processing'''
    feature_iit, classes = acculumate_feature(net, train_all_loader, 1)
    # print(feature_iit)
    tf_idf_map = {}
    #calculate_cp(features: dict, classes: list, dataset: str, coe: int, unlearn_class: int, tf_idf_map: dict):

    importance = calculate_cp(feature_iit, classes, FL_params.data_name, 0, FL_params.unlearn_class,tf_idf_map=tf_idf_map)
    # feature, score = importance.items()
    for feature, score in importance.items():
        # feature= feature.item()
        # score= score.item()
        print(f"Feature: {feature}, Importance: {score}")
    redundant_classes_indices = np.argsort(score)[-10:] #返回数值最大的10个index
    # 将tensor转换为整数类型
    redundant_classes_indices = redundant_classes_indices.cpu().numpy().astype(int)

    print('Classes indices: ', redundant_classes_indices)
    # max_index = np.argmax(importance)
    max_index = np.argmax(redundant_classes_indices)
    # print("最大值的索引：", max_index)

    # tf_idf_map = calculate_cp(feature_iit, classes, FL_params.data_name, 0, _idxforget_client_idx=FL_params.forget_client_idx)
    # print(tf_idf_map)
    threshold = get_threshold_by_sparsity(tf_idf_map, FL_params.sparsity)
    print('threshold', threshold)

    # trainset_todo = MyDataset(feature_iit, classes)
    # 获取类别名称
    all_classes = np.array(trainset.classes).tolist()
    redundant_classes = [all_classes[max_index]]
    FL_params.unlearn_class=max_index
    print("Class to be removed: ", redundant_classes)
    # all_classes.remove(max_index)
    # 使用列表推导式去除重复类别
    filtered_classes = [c for c in all_classes if c not in redundant_classes]

    # 更新 all_classes
    all_classes = filtered_classes
    print("PPPPPPPPPPleaseFiltered classes: ", len(all_classes))
    ################################

    '''test before pruning'''

    # 创建数据加载器
    # generate: def generate(dataset, list_classes: list):
    unlearn_testset=generate(testset, redundant_classes)
    rest_trainset=generate(testset, all_classes)


    # 创建自定义数据集对象
    custom_unlearn_dataset = CustomDataset(unlearn_testset, num_samples=100)
    custom_rest_dataset = CustomDataset(rest_trainset, num_samples=100)

    # 创建 DataLoader 对象
    unlearn_data_loader = DataLoader(custom_unlearn_dataset, batch_size=64, shuffle=True)
    rest_data_loader = DataLoader(custom_rest_dataset, batch_size=64, shuffle=True)

    # unlearn_testloader = torch.utils.data.DataLoader(unlearn_testset, batch_size=64, shuffle=True)
    # rest_testloader = torch.utils.data.DataLoader(rest_trainset, batch_size=64, shuffle=False)

    # for c in redundant_classes:
    #     all_classes.remove(c)
    # print('all classes: ', all_classes)

    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)

    device_cpu = torch.device("cpu")
    net.to(device_cpu)
    test(net, unlearn_data_loader) #unlearn_testloader
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(net, rest_data_loader)
    print('*' * 40)
    return max_index


def unlearning(mode, old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    """
    Parameters
    ----------
    old_global_models : list of DNN models
        In standard federated learning, all the global models from each round of training are saved.
    old_client_models : list of local client models
        In standard federated learning, the server collects all user models after each round of training.
    client_data_loaders : list of torch.utils.data.DataLoader
        This can be interpreted as each client user's own data, and each Dataloader corresponds to each user's data
    test_loader : torch.utils.data.DataLoader
        The loader for the test set used for testing
    FL_params : Argment（）
        The parameter class used to set training parameters

    Returns
    -------
    forget_global_model : One DNN model that has the same structure but different parameters with global_moedel
        DESCRIPTION.

    """

    if (FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')

    # if (not (FL_params.forget_client_idx in range(FL_params.N_client))):
    #     raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
    #         range(FL_params.N_client)))
    if (FL_params.unlearn_interval == 0 or FL_params.unlearn_interval > FL_params.global_epoch):
        print(FL_params.unlearn_interval)
        print(FL_params.global_epoch)
        raise ValueError(
            'FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')

    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)

    # print('*'*8, '尝试忘记某一个client，使用unlearn data和rest data', '*'*8)
    #TODO 其实是应该忘记某个client中的一个类
    # if (mode=='client'):
    unlearn_class_pruned = FL_params.unlearn_class
    print('unlearn_class_pruned ', unlearn_class_pruned)

    # 计算每个全局周期的起始和结束索引
    start_indices = range(0, len(old_client_models), FL_params.N_client)
    end_indices = range(FL_params.N_client, len(old_client_models) + 1, FL_params.N_client)

    # 遍历全局周期
    for start_idx, end_idx in zip(start_indices, end_indices):
        temp = old_client_models[start_idx:end_idx]  # 获取当前全局周期的模型列表
        if unlearn_class_pruned < len(temp):
            temp.pop(unlearn_class_pruned)  # 删除 forget_client 对应的模型
            old_client_models.extend(temp)  # 添加剩余的模型到列表
        # else:
        #     print("Error: forget_client index out of range")

    # 保留最后 FL_params.global_epoch 个全局周期的模型
    # old_client_models = old_client_models[-FL_params.global_epoch:]
    # print('old_client_models ', old_client_models)
    # old_client_models = [old_client_models[i:i + FL_params.N_client] for i in range(-FL_params.global_epoch, 0)]
    # print('old_client_models: ', old_client_models)

    for ii in range(FL_params.global_epoch):
        temp = list(old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client])
        # print('temp lens here', len(temp))
        if temp and unlearn_class_pruned < len(temp):
            for i in range(unlearn_class_pruned + 1, len(temp)):
                temp[i - 1] = temp[i]
            temp.pop(unlearn_class_pruned)
            # print('temp lens', len(temp))
        # else:
        #     print("Error: forget_client index out of range or temp list is empty")
        # print('temp', len(temp))
        # temp.pop(forget_client)  # During Unlearn, the model saved by the forgotten user pops up
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    # print('len of old cm: ', len(old_client_models)) #20

    GM_intv = np.arange(0, FL_params.global_epoch, FL_params.unlearn_interval, dtype=np.int16())
    # print('GM_intvhere', GM_intv) #[0-19]
    CM_intv = GM_intv - 1
    CM_intv = CM_intv[1:]
    # print('GM_old',old_global_models)
    selected_GMs = [old_global_models[ii:1] for ii in GM_intv]
    # print('len of selected GMs: ', len(selected_GMs)) #20
    for ii in GM_intv:
        if ii < len(old_global_models):
            selected_GMs.append(old_global_models[ii:ii+1])
            print('你能有一次吗', len(selected_GMs)) #21
        # else:
        #     print(f"Index {ii} out of range for old_global_models.")
    print('test', len(old_global_models))

    selected_CMs = [old_client_models[jj] for jj in CM_intv]
    print('len of slected CMs: ', len(selected_CMs)) #19

    """1. First, complete the model overlay from the initial model to the first round of global train"""
    """
    Since the inIT_model does not contain any information about the forgotten user at the start of the FL training, you just need to overlay the local Model of the other retained users, You can get the Global Model after the first round of global training.
    """
    epoch = 0
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))

    # From the paper: "It should be noticed that FedEraser can directly update the global model without calibration of the remaining clients' parameters at the first reconstruction epoch."
    # Note by Karly: I think something wrong with line153-154
    new_global_model = fedavg(selected_CMs[epoch])
    unlearn_global_models.append(copy.deepcopy(new_global_model))
    # print("unlearning--Federated Unlearning Global Epoch  = {}".format(epoch))

    """2. Then, the first round of global model as a starting point, the model is gradually corrected"""
    """
    In this step, the global Model obtained from the first round of global training was used as the new starting point for training, and a small amount of training was carried out with the data of the reserved user (a small amount means reducing the local epoch, i.e. Reduce the number of local training rounds for each user. The parameter forget_local_epoch_ratio is to control and reduce the number of local training rounds.) Gets the direction of iteration of the local Model parameter for each reserved user, starting with new_global_model.Note that this part of the user model is ref_client_models.

    Then we use the old_client_models and old_global_models saved from the unforgotten FL training, and the ref_client_models and new_global_Model that we get when we forget a user,To build the global model for the next round


    (ref_client_models - new_global_model) / ||ref_client_models - new_global_model||，Indicates the direction of model parameter iteration starting with a new global model that removes a user.Mark the direction as step_direction

    ||old_client_models - old_global_model||，Indicates the step size of the model parameter iteration starting with the old global model with a user removed.Step step_length

    So, the final direction of the new reference model is step_direction*step_length + new_global_model。
    """
    """
    Intuitive explanation of this part: Usually in IID data, after the data is sharded, the direction of model parameter iteration is roughly the same.The basic idea is to take full advantage of the client-model parameter data saved in standard FL training, and then, by correcting this part of the parameter, apply it to the iteration of the new global model that forgets a user.

    For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
    for unlearning FL：newGM_t-->newCM0, newCM1, newCM2, newCM3--> newGM_t+1
    oldGM_t and newGM_t essentially represents a different starting point for training. However, under the IID data, oldCM and newCM should converge in roughly the same direction.
    Therefore, we get newCM by using newCM-newGM_t as the starting point and training fewer rounds on user data, and then using (newCM-newGM_t)/|| newCM-newGM_t || as the current forgetting setting,
    Direction of model parameter iteration.Take || oldCM-oldGM_t || as the iteration step, and finally use || oldCM-oldGM_t ||*(newCM-newGM_t)/|| newCM-newGM_t || for the iteration of the new model.
    FedEraser iterative formula: newGM_t+1 = newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    """


    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    # New local_epoch for unlearning with FedEraser
    FL_params.local_epoch = np.ceil(FL_params.local_epoch * FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    # New global_epoch for unlearning with FedEraser
    FL_params.global_epoch = CM_intv.shape[0]

    print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(FL_params.global_epoch):
        # From the paper: "It should be noticed that FedEraser can directly update the global model without calibration of the remaining clients' parameters at the first reconstruction epoch."
        if (epoch == 0):
            continue
        print("Unlearning -- Federated Unlearning Global Epoch  = {}".format(epoch))
        global_model = unlearn_global_models[epoch]
        # print('global model : {}'.format(global_model))
        # print('ORRRR',client_data_loaders,'RRR',test_loader)
        # 25,1个torch.utils.data.dataloader.DataLoader object at **

        new_client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)

        # core part of FedEraser
        new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch], global_model)
        unlearn_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch
    return unlearn_global_models


def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    """


    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model

    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()  # Model Params： oldCM - oldGM_t
    new_param_update = dict()  # Model Params： newCM - newGM_t

    new_global_model_state = global_model_after_forget.state_dict()  # newGM_t

    return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    print('old_client_models', len(old_client_models))
    print('new_client_models', len(new_client_models))
    new_client_models=new_client_models[:len(old_client_models)]#[0]
    assert len(old_client_models) == len(new_client_models)

    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]

        return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]

        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
        new_param_update[layer] /= (ii + 1)  # Model Params： newCM

        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]  # 参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]  # 参数： newCM - newGM_t

        step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / torch.norm(new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)

    return_global_model.state_dict(return_model_state)

    return return_global_model # .state_dict()
    # return forget_global_model


# def unlearning_without_cali(old_global_models, old_client_models, FL_params):
#     """
#
#
#     Parameters
#     ----------
#     old_client_models : list of DNN models
#         All user local update models are saved during the federated learning and training process that is not forgotten.
#     FL_params : parameters
#         All parameters in federated learning and federated forgetting learning
#
#     Returns
#     -------
#     global_models : List of DNN models
#         In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round
#
#     """
#     """
#     The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
#                  For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
#     The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
#     """
#     if (FL_params.if_unlearning == False):
#         raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
#
#     if (not (FL_params.forget_client_idx in range(FL_params.N_client))):
#         raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
#             range(FL_params.N_client)))
#     forget_client = FL_params.forget_client_idx
#
#     # for ii in range(FL_params.global_epoch):
#     #     temp = old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client]
#     #     # temp.pop(forget_client)
#     #     old_client_models.append(temp)
#     old_client_models = old_client_models[-FL_params.global_epoch:]
#
#     uncali_global_models = list()
#     uncali_global_models.append(copy.deepcopy(old_global_models[0]))
#     epoch = 0
#     uncali_global_model = fedavg(old_client_models[epoch])
#     uncali_global_models.append(copy.deepcopy(uncali_global_model))
#     print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))
#
#     """
#     new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)
#
#     For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
#     For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
#     For uncalibrated federated forgotten learning, the parameter update of the unforgotten user in standard federated learning is used to directly overlay the new global model to obtain the next round of new global model.
#     """
#     old_param_update = dict()  # (oldCM_t - oldGM_t)
#     return_model_state = dict()  # newGM_t+1
#
#     for epoch in range(FL_params.global_epoch):
#         if (epoch == 0):
#             continue
#         print("unlearning without cali--Federated Unlearning Global Epoch  = {}".format(epoch))
#
#         current_global_model = uncali_global_models[epoch]  # newGM_t
#         # current_client_models = old_client_models[epoch]  # oldCM_t
#         current_client_models = [copy.deepcopy(model) for model in old_client_models[epoch]]
#
#         # print('old_global_model lens!!!!!!!', old_global_models) #是两个resnet
#         old_global_model = old_global_models[epoch]  # oldGM_t
#         # print("global没有s",type(old_global_model))#<class 'model_initiation.ResNet'>
#         # print("有s", type(old_global_models))
#         # print("没有s", current_client_models)
#         print("有s", current_client_models[0])
#         global_model_before_forget = old_global_models[epoch]#old_GM_t
#
#         for layer in current_global_model.state_dict().keys():
#             # State variable initialization
#             old_param_update[layer] = 0 * current_global_model.state_dict()[layer]
#             return_model_state[layer] = 0 * current_global_model.state_dict()[layer]
#             for model in current_client_models:
#                 print(type(model))
#
#             for ii in range(len(current_client_models)):
#                 old_param_update[layer] += current_client_models[ii].state_dict()[layer]
#             # print("old_param_update[layer] 的数据类型：", old_param_update[layer].dtype)
#             # print("ii 的数据类型：", type(ii))
#             #TODO 加了一个整数//
#             old_param_update[layer] //= (ii + 1)  # oldCM_t
#
#             old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[
#                 layer]  # 参数： oldCM_t - oldGM_t
#
#             return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[
#                 layer]  # newGM_t + (oldCM_t - oldGM_t)
#
#         return_global_model = copy.deepcopy(old_global_models[0])
#         return_global_model.state_dict(return_model_state)
#
#         uncali_global_models.append(return_global_model)
#
#     return uncali_global_models
# def unlearning_without_cali(old_global_models, old_client_models, FL_params):
#     """
#
#
#     Parameters
#     ----------
#     old_client_models : list of DNN models
#         All user local update models are saved during the federated learning and training process that is not forgotten.
#     FL_params : parameters
#         All parameters in federated learning and federated forgetting learning
#
#     Returns
#     -------
#     global_models : List of DNN models
#         In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round
#
#     """
#     """
#     The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
#                  For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
#     The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
#     """
#     if (FL_params.if_unlearning == False):
#         raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
#
#     if (not (FL_params.forget_client_idx in range(FL_params.N_client))):
#         raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
#             range(FL_params.N_client)))
#     forget_client = FL_params.forget_client_idx
#
#     for ii in range(FL_params.global_epoch):
#         temp = old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client]
#         temp.pop(forget_client)
#         old_client_models.append(temp)
#     old_client_models = old_client_models[-FL_params.global_epoch:]
#
#     uncali_global_models = list()
#     uncali_global_models.append(copy.deepcopy(old_global_models[0]))
#     epoch = 0
#     uncali_global_model = fedavg(old_client_models[epoch])
#     uncali_global_models.append(copy.deepcopy(uncali_global_model))
#     print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))
#
#     """
#     new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)
#
#     For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
#     For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
#     For uncalibrated federated forgotten learning, the parameter update of the unforgotten user in standard federated learning is used to directly overlay the new global model to obtain the next round of new global model.
#     """
#     old_param_update = dict()  # (oldCM_t - oldGM_t)
#     return_model_state = dict()  # newGM_t+1
#
#     for epoch in range(FL_params.global_epoch):
#         if (epoch == 0):
#             continue
#         print("Federated Unlearning Global Epoch  = {}".format(epoch))
#
#         current_global_model = uncali_global_models[epoch]  # newGM_t
#         current_client_models = old_client_models[epoch]  # oldCM_t
#         old_global_model = old_global_models[epoch]  # oldGM_t
#         # global_model_before_forget = old_global_models[epoch]#old_GM_t
#
#         for layer in current_global_model.state_dict().keys():
#             # State variable initialization
#             old_param_update[layer] = 0 * current_global_model.state_dict()[layer]
#             return_model_state[layer] = 0 * current_global_model.state_dict()[layer]
#
#             for ii in range(len(current_client_models)):
#                 old_param_update[layer] += current_client_models[ii].state_dict()[layer]
#             old_param_update[layer] /= (ii + 1)  # oldCM_t
#
#             old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[
#                 layer]  # 参数： oldCM_t - oldGM_t
#
#             return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[
#                 layer]  # newGM_t + (oldCM_t - oldGM_t)
#
#         return_global_model = copy.deepcopy(old_global_models[0])
#         return_global_model.load_state_dict(return_model_state)
#
#         uncali_global_models.append(return_global_model)
#
#     return uncali_global_models
def unlearning_without_cali(old_global_models, old_client_models, FL_params):
    """
    Parameters
    ----------
    old_global_models : list
        List of old global models.
    old_client_models : list
        List of old client models.
    FL_params : object
        Parameters for federated learning.

    Returns
    -------
    uncali_global_models : list
        List of uncalibrated global models after unlearning.
    """
    # FL_params = Arguments()

    if not FL_params.if_unlearning:
        raise ValueError("FL_params.if_unlearning should be set to True if you want to unlearn with a certain user")

    # if FL_params.unlearn_class not in range(FL_params.N_client):
    #     raise ValueError("FL_params.forget_client_idx is not assigned correctly. "
    #                      "forget_client_idx should be in {}".format(range(FL_params.N_client)))

    unlearn_class_pruned = FL_params.unlearn_class

    # uncali_global_models = []

    # Remove forget_client's model from old_client_models
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii * FL_params.N_client: (ii + 1) * FL_params.N_client]
        if temp and unlearn_class_pruned < len(temp):
            temp.pop(unlearn_class_pruned)
        # else:
        #     print("Error: forget_client index out of range or temp list is empty")

        old_client_models.append(temp)

    # old_client_models = old_client_models[-FL_params.global_epoch:]
    uncali_global_models = old_global_models[0]

    # uncali_global_models.append(copy.deepcopy(old_global_models[0]))

    # Iterate over epochs for unlearning
    for epoch in range(FL_params.global_epoch):
        if epoch == 0:
            continue

        print("Federated Unlearning without Calibration Global Epoch  = {}".format(epoch))

        current_global_model = uncali_global_models#[epoch]
        # print('current_global_model TYpe lens', len(current_global_model))
        current_client_models = old_client_models[epoch]
        old_global_model = old_global_models[epoch]

        old_param_update = {}  # (oldCM_t - oldGM_t)
        return_model_state = {}  # newGM_t+1

        for layer in current_global_model.state_dict().keys():
            old_param_update[layer] = torch.zeros_like(current_global_model.state_dict()[layer])
            return_model_state[layer] = torch.zeros_like(current_global_model.state_dict()[layer])

            # Calculate the parameter update (oldCM_t - oldGM_t)
            old_param_update[layer] = current_client_models.state_dict()[layer]
            # for client_model in current_client_models:
            #     old_param_update[layer] += client_model.state_dict()[layer]
            # old_param_update[layer] /= len(current_client_models)

            old_param_update[layer] -= old_global_model.state_dict()[layer]

            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[layer]

        # Update the return_global_model with the computed state_dict
        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)

        uncali_global_models=return_global_model#.append(return_global_model)

    return uncali_global_models
