# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
from pathlib import Path

import total_variance as total_variance
#ourself libs
from model_initiation import model_init
from data_preprocess import data_set

from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain, test
from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity, select_least_important_clients


import torch
from torchvision import datasets, transforms

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

def Class_pruner(net, FL_params):
    # project_dir = Path(__file__).resolve().parent
    # model_path = project_dir / 'ckpt' / FL_params.model_name / FL_params.model_file
    # pruned_save_info = project_dir / 'ckpt' / 'pruned' / FL_params.model_name
    # finetuned_save_info = project_dir / 'ckpt' / 'finetuned' / FL_params.model_name

    models = []
    trainset, testset = data_set('cifar10')
    loaders = []
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)
    for _ in range(FL_params.N_client):
        models.append(net)
        loaders.append(train_loader)
    tf_idf = []
    for m, l in zip(models, loaders):
        features, classes = acculumate_feature(m, l, stop=10)
        # 计算TF-IDF
        tf_idf_map = calculate_cp(features, classes, dataset='cifar10', coe=1, unlearn_client=0)
        tf_idf.append(tf_idf_map)
    # cp_config = {"threshold": threshold, "map": tf_idf_map}
    print('tf_idf: ', tf_idf)
    least_important_clients = select_least_important_clients(tf_idf, num_clients_to_select=1)

    print("Selected least important clients:", least_important_clients)

    # config_list = [{
    #     'sparsity': FL_params.sparsity,
    #     'op_types': ['Conv2d']
    # }]
    # # 设置剪枝配置
    # cp_config = {
    #     "threshold": get_threshold_by_sparsity(tf_idf_map, sparsity=0.5),
    #     "map": tf_idf_map
    # }
    # # 初始化并使用TFIDFPruner
    # pruner = TFIDFPruner(model=models[0], config_list=config_list, cp_config=cp_config)
    # pruner.update_masker(models[0], cp_config["threshold"], tf_idf_map)
    # pruner.compress()
    # pruned_model_path = pruned_save_info / ('seed_' +
    #                                         time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) +
    #                                         '_model.pth')
    # pruned_mask_path = pruned_save_info / ('seed_' +
    #                                        time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) +
    #                                        '_mask.pth')
    # pruner.export_model("pruned_model.pth", "pruned_mask.pth")
    #
    # pruned_net = model_init('cifar10', 'cpu')
    # # pruned_net.cuda()
    # load_model_pytorch(pruned_net, pruned_model_path, FL_params.model_name)
    # print('*'*8,'Finished pruning model','*'*8)
    return least_important_clients, net

    # '''load data and model'''
    # if FL_params.data_name == 'cifar10':
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     net = model_init('cifar10', device)
    #     trainset, testset = data_set('cifar10')
    #     total_classes = 10 # [0-9]
    # elif FL_params.data_name == 'mnist':
    #     trainset, testset=data_set('mnist')
    #     # net = model_init('mnist', FL_params.model_name)
    #     total_classes = 10  # [0-9]
    # elif FL_params.data_name == 'cifar100':
    #     trainset,testset=data_set('cifar100')
    #     # net = model_init('cifar100', FL_params.model_name)
    #     total_classes = 10  # [0-9]
    # elif FL_params.data_name == 'adult':
    #     trainset,testset=data_set('adult')
    #     # net = model_init('adult', FL_params.model_name)
    #     total_classes = 2  # [0-1]
    # elif FL_params.data_name == 'purchase':
    #     trainset,testset=data_set('purchase')
    #     # net = model_init('purchase', FL_params.model_name)
    #     total_classes = 2  # [0-9]
    #
    # # 将tensor转换为整数类型
    # if (FL_params.fats_method == 'sample'):
    #     print('#'*5,'Trying to find redundant sample index:','#'*5)
    #     train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)
    #
    #     '''pre-processing'''
    #     feature_iit, classes = acculumate_feature(net, train_all_loader, 1, FL_params)
    #     tf_idf_map = calculate_cp(feature_iit, classes, FL_params.data_name, 0, FL_params.forget_client_idx)
    #     # threshold = get_threshold_by_sparsity(tf_idf_map, FL_params.sparsity)
    #     # print('threshold', threshold)
    #
    #     # print('Importance feature_iit',importance,feature_iit, classes, FL_params.forget_client_idx, tf_idf_map)
    #     for feature, score in tf_idf_map.items():
    #         if (FL_params.data_name == 'cifar10' or FL_params.data_name == 'mnist'):
    #             redundant_classes_indices = np.argsort(score)[-10:]  # 返回数值最大的10个index
    #         elif (FL_params.data_name == 'purchase' or FL_params.data_name == 'adult'):
    #             redundant_classes_indices = np.argsort(score)[-2:]
    #         elif (FL_params.data_name == 'cifar100'):
    #             redundant_classes_indices = np.argsort(score)[-100:]
    #
    #     redundant_classes_indices = redundant_classes_indices.cpu().numpy().astype(int)
    #
    #     print('Classes indices: ', redundant_classes_indices)
    #     max_index = np.argmax(redundant_classes_indices)
    #     # print("最大值的索引：", max_index)
    #     print('tf_idf', tf_idf_map)
    #     threshold = get_threshold_by_sparsity(tf_idf_map, FL_params.sparsity)
    #     print('threshold', threshold)
    #
    #     # trainset_todo = MyDataset(feature_iit, classes)
    #     # 获取类别名称
    #     all_classes = np.array(trainset.classes).tolist()
    #     print('all_classes', all_classes)
    #     redundant_classes = [all_classes[max_index]]
    #     # FL_params.forget_client_idx=max_index
    #     print("Class to be removed: ", redundant_classes)
    #     # all_classes.remove(max_index)
    #     # 使用列表推导式去除重复类别
    #     filtered_classes = [c for c in all_classes if c not in redundant_classes]
    #
    #     # 更新 all_classes
    #     all_classes = filtered_classes
    #     print("#Remain classes: ", len(all_classes))
    #     return max_index, net
    # elif(FL_params.fats_method=='client'):
    #     # print('Clinet Pruning HERE: trainset', trainset)
    #     train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)
    #
    #     '''pre-processing'''
    #     feature_iit, client_list = acculumate_feature(net, train_all_loader, 1, FL_params)
    #     tf_idf_map = calculate_cp(feature_iit, client_list, FL_params.data_name, 0, FL_params.forget_client_idx)
    #     # print('CLIENT tf_idf_map', tf_idf_map)
    #     # print('CLIENT LIST', client_list)
    #
    #     redundant_classes_indices=[]
    #
    #     for feature, score in tf_idf_map.items():
    #         print(f"Feature: {feature}, Score length: {len(score)}, Scores: {score}")
    #
    #         max_score_index = torch.argmax(score).item()
    #         redundant_classes_indices.append(max_score_index)
    #         print("Max score index for feature {}: {}".format(feature, max_score_index))
    #
    #         # score_indices = np.argsort(score)[-FL_params.K:]
    #         # if score_indices.size == 1:
    #         #     redundant_classes_indices.append(int(score_indices.item()))  # Correct way to convert a single-element tensor to an integer
    #         #     print("CLIENT Redundant Index:", redundant_classes_indices)
    #         # else:
    #         #     print("Error: score_indices contains more than one element.")
    #         #     # Handle the situation, e.g., by selecting the appropriate element:
    #         #     redundant_classes_indices.append(int(score_indices[0].item()))  # Example: Choosing the first element
    #         #     print("Chosen index:", redundant_classes_indices)
    #         score_indices = torch.argsort(score, descending=True)[:FL_params.K]
    #         chosen_index = score_indices[0].item()  # Take the first of the top K scores
    #         redundant_classes_indices.append(chosen_index)
    #         print("Chosen client feature {}: {}".format(feature, chosen_index))
    #
    #
    #
    #
    #     # all_redundant_indic es = []
    #     # for feature, score in tf_idf_map.items():
    #         # all_redundant_indices.append(np.argsort(score)[-FL_params.K:])
    #
    #
    #     # redundant_classes_indices = redundant_classes_indices.cpu().numpy().astype(int)
    #
    #
    #     print('Client indices: ', redundant_classes_indices)
    #     # for i in range(len(redundant_classes_indices)):
    #     #     max_in_two=max(redundant_classes_indices[i])
    #     #     index=redundant_classes_indices[max_in_two]
    #     max_index = max(redundant_classes_indices)
    #     print('Max index: ', max_index),type(max_index)
    #     return max_index, net

# def some_condition_to_identify_Xu(image, label):
#     """
#     识别是否应该遗忘特定的图像样本。
#
#     参数:
#     image (torch.Tensor): CIFAR-10数据集中的图像张量。
#     label (int): 与图像相对应的类别标签。
#
#     返回:
#     bool: 如果图像应该被遗忘，则返回True；否则返回False。
#     """
#     return label == label_to_remove


def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    
    
    # all_global_models, all_client_models 为保存起来所有的old FL models
    print(5*"#"+"  Federated Learning Start"+5*"#")
    std_time = time.time()
    old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = (std_time - end_time)
    
    print(5*"#"+"  Federated Learning End"+5*"#")
    
    
    print('\n')
    """4.2 unlearning  a client，Federated Unlearning"""
    print(5*"#"+"  Federated Unlearning Start  "+5*"#")
    std_time = time.time()
    #Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
    FL_params.if_unlearning = True
    #Set the parameter forget_client_IDx to mark the user's IDX that needs to be forgotten
    # FL_params.forget_client_idx = 2
    #TODO
    # unlearn_GMs = unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)

    if (FL_params.fats_method == 'sample'):
        print('#'*4, 'Sample-level Unlearning', '#'*4)
        unlearn_GMs = total_variance.sample_level_unlearning(old_GMs, old_CMs, client_loaders, test_loader, 3,  FL_params)

    elif (FL_params.fats_method == 'client'):
        print('#' * 4, 'Client-level Unlearning', '#' * 4)
        FL_params.forget_client_idx = 2
        unlearn_GMs, train_acc, train_loss, test_acc, test_loss= total_variance.client_level_unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)

        # unlearn_GMs = total_variance.client_level_unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)

    end_time = time.time()
    time_unlearn = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning End  "+5*"#")

    
    print('\n')
    """4.3 unlearning a client，Federated Unlearning without calibration"""
    print(5*"#"+"  Federated Unlearning without Calibration Start  "+5*"#")
    std_time = time.time()
    uncali_unlearn_GMs = unlearning_without_cali(old_GMs, old_CMs, FL_params)
    end_time = time.time()
    time_unlearn_no_cali = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning without Calibration End  "+5*"#")
    
    print(" Learning time consuming = {} secods".format(-time_learn))
    print(" Unlearning time consuming = {} secods".format(-time_unlearn)) 
    print(" Unlearning no Cali time consuming = {} secods".format(-time_unlearn_no_cali))
    
    
    return old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs

    
    
    
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

def unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, t_u, k_u,FL_params):
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
    r_u = ((t_u - 1) // FL_params.local_epoch) + 1
    
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
        
    if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    if(FL_params.unlearn_interval == 0 or FL_params.unlearn_interval >FL_params.global_epoch):
        raise ValueError('FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    
    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)

    # 计算要遗忘的客户端索引
    k_u = FL_params.forget_client_idx
    # 从每轮训练中移除要遗忘的客户端模型
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        temp.pop(k_u)#During Unlearn, the model saved by the forgotten user pops up
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    
    GM_intv = np.arange(0,FL_params.global_epoch+1, FL_params.K, dtype=np.int16())
    CM_intv  = GM_intv - FL_params.K
    CM_intv = CM_intv[FL_params.K:]
    
    selected_GMs = [old_global_models[ii] for ii in GM_intv]
    selected_CMs = [old_client_models[jj] for jj in CM_intv]
    
    
    """1. First, complete the model overlay from the initial model to the first round of global train"""
    """
    Since the inIT_model does not contain any information about the forgotten user at the start of the FL training, you just need to overlay the local Model of the other retained users, You can get the Global Model after the first round of global training.
    """
    epoch = 0
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))

    # for epoch in range(1, len(selected_GMs)):
    #     current_global_model = unlearn_global_models[-1]
    #     ref_global_model = selected_GMs[epoch]
    #
    #     # 对每个保留的客户端进行一次全局训练
    #     ref_client_models = global_train_once(current_global_model, client_data_loaders, test_loader, FL_params)
    #
    #     # 执行遗忘步骤
    #     new_global_model = unlearning_step_once(selected_CMs[epoch - 1], ref_client_models, ref_global_model,
    #                                             current_global_model)
    #     unlearn_global_models.append(copy.deepcopy(new_global_model))
    #     print(f"Federated Unlearning Global Epoch = {epoch}")
    
    new_global_model = fedavg(selected_CMs[epoch])
    unlearn_global_models.append(copy.deepcopy(new_global_model))
    print("Federated Unlearning Global Epoch  = {}".format(epoch))
    
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
    for unblearning FL：newGM_t-->newCM0, newCM1, newCM2, newCM3--> newGM_t+1
    oldGM_t and newGM_t essentially represents a different starting point for training. However, under the IID data, oldCM and newCM should converge in roughly the same direction.
    Therefore, we get newCM by using newcm-newgm_t as the starting point and training fewer rounds on user data, and then using (newcm-newgm_t)/|| newcm-newgm_t || as the current forgetting setting,
    Direction of model parameter iteration.Take || oldcm-oldgm_t || as the iteration step, and finally use || oldcm-oldgm_t ||*(newcm-newgm_t)/|| newcm-newgm_t |0 |1 for the iteration of the new model.
    FedEraser iterative formula: newGM_t+1 = newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    
    """
    

    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    FL_params.global_epoch = CM_intv.shape[0]
    
    
    print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        global_model = unlearn_global_models[epoch]

        new_client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)

        new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch+1], global_model)

        unlearn_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch
    return unlearn_global_models #[-1]


    
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
    old_param_update = dict()#Model Params： oldCM - oldGM_t
    new_param_update = dict()#Model Params： newCM - newGM_t
    
    new_global_model_state = global_model_after_forget.state_dict()#newGM_t
    
    return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    # print("lai",len(old_client_models),len(new_client_models))
    assert len(old_client_models) == len(new_client_models)
    
    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        
        return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii+1)#Model Params： oldCM
        new_param_update[layer] /= (ii+1)#Model Params： newCM
        
        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]#参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]#参数： newCM - newGM_t
        
        step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
        
        return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction
    
    
    
    
    
    return_global_model = copy.deepcopy(global_model_after_forget)
    
    return_global_model.load_state_dict(return_model_state)
    
    return return_global_model


    # return forget_global_model
    
    
def unlearning_without_cali(old_global_models, old_client_models, FL_params):
    """
    

    Parameters
    ----------
    old_client_models : list of DNN models
        All user local update models are saved during the federated learning and training process that is not forgotten.
    FL_params : parameters
        All parameters in federated learning and federated forgetting learning

    Returns
    -------
    global_models : List of DNN models
        In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round

    """
    """
    The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
                 For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
    The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
    """
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
    print('FL_params.forget_client_idx: ', FL_params.forget_client_idx)

    # if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
    #     raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    forget_client = FL_params.forget_client_idx
    
    
    for epoch in range(FL_params.global_epoch):
        # print('old_client', len(old_client_models))
        print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))

        temp = old_client_models[epoch*FL_params.N_client: epoch*FL_params.N_client+FL_params.N_client]
        # print('temp',len(temp))、
        temp.pop(forget_client)
        # print('#' * 4, 'Remove {} clients datapoint from global'.format(forget_client), '#' * 4)
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    uncali_global_models = list()
    # uncali_global_models.append(copy.deepcopy(old_global_models[-1]))

    epoch = 0
    for epoch in range(FL_params.local_epoch):
        uncali_global_model = fedavg(old_client_models[epoch])
        uncali_global_models.append(copy.deepcopy(uncali_global_model))
    # print('uncali_global_models: ', len(uncali_global_models))

    """
    new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)
    
    For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
    For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
    For uncalibrated federated forgotten learning, the parameter update of the unforgotten user in standard federated learning is used to directly overlay the new global model to obtain the next round of new global model.
    """
    old_param_update = dict()#(oldCM_t - oldGM_t)
    return_model_state = dict()#newGM_t+1
    
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))

        current_global_model = uncali_global_models[epoch]#newGM_t
        current_client_models = old_client_models[epoch]#oldCM_t
        old_global_model = old_global_models[epoch]#oldGM_t
        # global_model_before_forget = old_global_models[epoch]#old_GM_t


        for layer in current_global_model.state_dict().keys():
            #State variable initialization
            old_param_update[layer] = 0*current_global_model.state_dict()[layer]
            return_model_state[layer] = 0*current_global_model.state_dict()[layer]

            for ii in range(len(current_client_models)):
                old_param_update[layer] += current_client_models[ii].state_dict()[layer]
            old_param_update[layer] /= (ii+1)# oldCM_t

            old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[layer]#参数： oldCM_t - oldGM_t

            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[layer]#newGM_t + (oldCM_t - oldGM_t)

        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)

        uncali_global_models.append(return_global_model)

    return uncali_global_models
    
    
    
    
    

    
    



























