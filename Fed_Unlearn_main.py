# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
# %%
import torch
import time
from datetime import datetime
import argparse
import os
import numpy as np
import csv
import copy
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchtext
from torchvision import datasets, transforms

# ourself libs
from data_preprocess import splitExpertData, splittExpertModel, load_and_partition_data, create_models
from Fed_moe_base import train_clients, test_clients, mix_train_clients, mix_train_clients_step2, \
    membership_inference_attack, quick_retrain, read_data
from sample_data import create_user_dataloaders
from Models import CNNCifar, CNNFashion, Net_purchase, Net_adult, save_model
from FL_base import test
from LanguageModels import RNNGate, RNNTextClassifier

"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.multi_model = True
        self.N_client = 10
        self.data_1 = 'cifar'  # 'fashion' 'purchase' 'adult'
        self.data_2 = 'mnist'  # 'fashion' 'adult' 'purchase'

        self.all_classes = 0
        self.unlearn_class = [1]
        self.unlearn_class_tag=None
        self.unlearn_class_num = 2

        self.model_1 = 'cnn_cifar'  # 'cnn_fashion' 'purchase' 'adult' 'cnn_cifar'
        self.model_2 = 'cnn_fashion'  # 'cnn_fashion' 'adult' 'purchase'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Federated Unlearning Settings
        self.forget_client_idx = [1]  # If want to forget, change None to the client index
        self.forget_clients_num = 1  # 1-10

        self.global_epoch = 1500  # 20#1500#0  # 600#20#600  # T
        self.local_epoch = 100 #100  # 3  # 10#2#10 # E

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()

        self.if_retrain = True  # False
        self.p = 0.7
        self.opt = 0
        self.opt_out = [1]
        self.opt_out_class=None
        self.n_data = 100
        self.n_data_test = 200
        self.n_data_val = 200
        self.overlap = True
        self.frac = 0.1
        self.train_gate_only = False
        self.iid = False
        self.freeze = False
        self.seperate_mix = False
        self.alpha = 0
        self.users_list_2nd = []
        self.users_list_1st = []
        self.type_count_1st = 0
        self.type_count_2nd = 0

        self.reserve_client_list = []
        self.opt_in = []
        self.temp_forget_idx = []
        self.attack_1 = []
        self.attack_2 = []
        self.data_class = 0
        self.mode = -1  # -1
        self.user_list={}

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Federated Unlearning Arguments')
        parser.add_argument('--unlearn_class_num', type=int, default=self.unlearn_class_num,
                            help='Number of clients to forget')
        parser.add_argument('--global_epoch', type=int, default=self.global_epoch, help='Number of global epochs')
        parser.add_argument('--local_epoch', type=int, default=self.local_epoch, help='Number of local epochs')
        parser.add_argument('--data_1', type=str, default=self.data_1, help="name of dataset1")
        parser.add_argument('--data_2', type=str, default=self.data_2, help="name of dataset2")
        parser.add_argument('--opt_out', type=int, default=self.opt_out, help='User defined forget client')

        parser.add_argument('--opt', type=float, default=0.5, help='fraction of clients that opt-in (default: 0.5)')
        parser.add_argument('--p', type=float, default=self.p, help='majority class percentage (default: 0.3)')
        parser.add_argument('--n_data', type=float, default=100, help="datasize on each client")
        parser.add_argument('--n_data_test', type=float, default=200, help="test datasize on each client")
        parser.add_argument('--n_data_val', type=float, default=200, help="validation datasize on each client")
        parser.add_argument('--overlap', action='store_true',
                            help='whether to allow label overlap between clients or not')
        parser.add_argument('--iid', action='store_false', help='whether i.i.d or not')

        parser.add_argument('--model_1', type=str, default=self.model_1, help='which model to use')
        parser.add_argument('--model_2', type=str, default=self.model_2, help='which model to use')
        parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients")

        parser.add_argument('--N_client', type=int, default=10, help="number of clients")
        parser.add_argument('--local_lr', type=float, default=0.01, help="learning rate")
        parser.add_argument('--alpha', type=float, default=0.0,
                            help='value that controls dirichlet parameter for data size skew.')
        parser.add_argument('--mode', type=int, default=self.mode, help='4 modes:1,2,3,4')

        args = parser.parse_args()

        self.alpha = args.alpha
        self.unlearn_class_num = args.unlearn_class_num
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch

        self.opt = args.opt
        self.p = args.p
        self.n_data = args.n_data
        self.n_data_test = args.n_data_test
        self.n_data_val = args.n_data_val
        self.overlap = args.overlap
        self.model_1 = args.model_1
        self.model_2 = args.model_2
        self.frac = args.frac
        self.N_client = args.N_client
        self.local_lr = args.local_lr
        self.data_1 = args.data_1
        self.data_2 = args.data_2
        self.opt_out = args.opt_out
        self.iid = args.iid
        self.mode = args.mode


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    FL_params.parse_args()
    torch.manual_seed(FL_params.seed)
    print("device:", FL_params.device)

    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_1 +" and "+FL_params.data_2+ (
        " for our Federated Unlearning experiment.\n"))
    print('We are going to forget', FL_params.unlearn_class_num, 'class, with p=', FL_params.p, ' and the opt out client is: ',
          FL_params.opt_out)

    writer = SummaryWriter(comment=f'lr_{FL_params.local_lr}_p_{FL_params.p}_opt_{FL_params.opt}')

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")

    # Generate six different types of user lists, each containing 100 items
    type_1_users = list(range(100))
    type_2_users = list(range(100, 200))
    type_3_users = list(range(200, 300))
    type_4_users = list(range(300, 400))
    type_5_users = list(range(400, 500))
    type_6_users = list(range(500, 600))
    type_7_users = list(range(600, 700))

    # Combine all types into a dictionary for easy access
    user_types_all = {
        "res_cifar10": type_1_users,
        "cnn_cifar10": type_2_users,
        "cnn_mnist": type_3_users,
        "cnn_fashion": type_4_users,
        "fcn_purchase": type_5_users,
        "fcn_adult": type_6_users,
        "agnews":type_7_users
    }
    user_types = {
        "res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []
    }

    # Set the number of clients to select from each type
    selected_users_count = FL_params.N_client

    # Randomly select clients across all types to form a total of selected_users_count
    selected_users = []
    for _ in range(selected_users_count):
        # Randomly pick a type
        user_type_key = random.choice(list(user_types_all.keys()))
        # Randomly pick a user from the selected type list
        if len(user_types_all[user_type_key]) > 0:
            selected_user = random.choice(user_types_all[user_type_key])
            selected_users.append(selected_user)
            # Remove the selected user from the list to avoid duplicate selections
            user_types_all[user_type_key].remove(selected_user)
            user_types[user_type_key].append(selected_user)

    print("Selected users:", selected_users, user_types)
    FL_params.user_list = user_types
    datasets_train, datasets_test, datasets_users, datasets_users_val, datasets_users_test=[],[],[],[],[]
    client_glob_model = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    client_gate_model = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    client_local_model={"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    for user_type, users in user_types.items():
        # If there are multiple values for the key (multiple users)
        for user in users:

            data_train, data_test, data_users, data_users_val, data_users_test = load_and_partition_data(
                user_type, len(users), FL_params)
            if len(users) > 0:
                datasets_train.append(data_train)
                datasets_test.append(data_test)
                datasets_users.append(data_users)
                datasets_users_val.append(data_users_val)
                datasets_users_test.append(data_users_test)

            # Create models for the current user
            net_glob, gate_model, net_local = create_models(user_type, FL_params)
            if len(users) > 0:
                if user_type not in client_glob_model:
                    client_glob_model[user_type] = []
                if user_type not in client_gate_model:
                    client_gate_model[user_type] = []
                if user_type not in client_local_model:
                    client_local_model[user_type] = []

                # Append models for the current user type
                client_glob_model[user_type].append(net_glob)
                client_gate_model[user_type].append(gate_model)
                client_local_model[user_type].append(net_local)

    #     print(f"Processed user type: {user_type}")
    # print('client_glob_model',len(client_glob_model),'client_gate_model',len(client_gate_model),'client_local_model',len(client_local_model))
    # print('client_local_model',client_local_model.keys(),client_local_model.values())
    # Creating DataLoaders for Six Different Types
    client_loaders = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    client_val_loaders = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    client_test_loaders = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}

    # Iterate through the six types and create DataLoaders for each type
    for key, user_list in FL_params.user_list.items():
        if len(user_list) > 0:
            idx = list(FL_params.user_list.keys()).index(key)
            client_loaders[key].append(create_user_dataloaders(datasets_train[idx], datasets_users[idx],
                                                          batch_size=FL_params.local_batch_size))
            client_val_loaders[key].append(create_user_dataloaders(datasets_train[idx], datasets_users_val[idx],
                                                              batch_size=FL_params.local_batch_size))
            client_test_loaders[key].append(create_user_dataloaders(datasets_test[idx], datasets_users_test[idx],
                                                               batch_size=FL_params.local_batch_size))

    # Initialize test_loader
    combined_test_loader = {"res_cifar10": [],
        "cnn_cifar10": [],
        "cnn_mnist": [],
        "cnn_fashion": [],
        "fcn_purchase": [],
        "fcn_adult": [],
        "agnews": []}
    for key,value in FL_params.user_list.items():
        combined_test_loader[key].extend(value)

    # Updating new_list for all client types
    new_list = []
    counter = 0
    for sublist in FL_params.user_list.values():
        new_sublist = []
        for _ in sublist:
            new_sublist.append(counter)
            counter += 1
        new_list.append(new_sublist)

    # Defining idxs_users for all types
    idxs_users = {key: list(range(len(user_list))) for key, user_list in FL_params.user_list.items()}

    # Federated Learning and Unlearning Training
    net_mix_glob_fedAvg = []
    for key in FL_params.user_list.keys():
        if key == 'res_cifar10':
            print('-------Using pre-trained model for resnet cifar10------')
            net_glob_fedAvg = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
            gm_checkpoint = torch.load('1500multi_cifar_global.pth', map_location='cpu')
        elif key == 'cnn_cifar10':
            print('-------Using pre-trained model for cnn cifar------')
            net_glob_fedAvg = CNNCifar(FL_params).to(FL_params.device)
            gm_checkpoint = torch.load('1500cifar_cnn_global.pth', map_location='cpu')
        elif key == 'fcn_purchase':
            print('-------Using pre-trained model for purchase------')
            net_glob_fedAvg = Net_purchase().to(FL_params.device)
            gm_checkpoint = torch.load('1500_purchase_global.pth', map_location='cpu')
        elif key == 'fcn_adult':
            print('-------Using pre-trained model for adult------')
            net_glob_fedAvg = Net_adult().to(FL_params.device)
            gm_checkpoint = torch.load('1500_adult_global.pth', map_location='cpu')
        elif key == 'cnn_mnist':
            print('-------Using pre-trained model for mnist------')
            net_glob_fedAvg = CNNFashion(FL_params).to(FL_params.device)
            gm_checkpoint = torch.load('1500multi_mnist_global.pth', map_location='cpu')
        elif key == 'cnn_fashion':
            print('-------Using pre-trained model for fashion mnist------')
            net_glob_fedAvg = CNNFashion(FL_params).to(FL_params.device)
            gm_checkpoint = torch.load('1500fashionmulti_mnist_global.pth', map_location='cpu')
        elif key == 'agnews':
            print('-------Using pre-trained model for agnews------')
            if len(client_glob_model[key])>0:
                net_glob_fedAvg = client_glob_model[key][-1]
                gm_checkpoint = torch.load('1500_agnews.pth', map_location='cpu')
        else:
            print(f"No pre-trained model available for key: {key}")
            continue

        net_glob_fedAvg.load_state_dict(gm_checkpoint, strict=False)
        net_mix_glob_fedAvg.append(net_glob_fedAvg)

    # Iterate over all six types to fine-tune models for each type
    for key in FL_params.user_list.keys():
        if len(client_gate_model[key]) > 0:
            std_time_finetune = time.time()

            finetuned, locals_nets, val_acc_ft, train_acc_ft, val_acc_locals, train_acc_locals, client, ft_time = train_clients(
                writer, key, idxs_users[key], datasets_train[list(FL_params.user_list.keys()).index(key)],
                datasets_test[list(FL_params.user_list.keys()).index(key)],
                datasets_users[list(FL_params.user_list.keys()).index(key)],
                datasets_users_val[list(FL_params.user_list.keys()).index(key)],
                datasets_users_test[list(FL_params.user_list.keys()).index(key)], net_glob_fedAvg,
                client_local_model[key][0],
                FL_params, client_loaders[key], client_val_loaders[key]
            )

            save_model(finetuned[0], "ckpt/finetuned",
                       f"finetuned_{key}_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")
            end_time_finetune = time.time()
            time_learn_finetune = (end_time_finetune - std_time_finetune)

            print(f" Fine-tuning for {key} time consuming = {time_learn_finetune} seconds")
            net_mix_glob_fedAvg.append(copy.deepcopy(finetuned[0]))
    exit(0)

    # total_users = FL_params.N_total_client
    # selected_users_count = FL_params.N_client
    #
    # # 随机选择10个用户
    # selected_users = random.sample(range(total_users), selected_users_count)
    # # selected_users = [[1,2],[3,4,5]]
    # # selected_users = [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]
    # FL_params.opt_in = selected_users
    #
    # users_type = []
    # each_user_count=[]
    #
    # # 随机选择使用CNN和ResNet的用户数量
    # FL_params.type_count_1st = random.randint(1, selected_users_count)
    # FL_params.type_count_2nd = selected_users_count - FL_params.type_count_1st
    # # print(FL_params.type_count_1st, FL_params.type_count_2nd)
    #
    # # 随机选择哪些用户使用CNN，哪些使用ResNet
    # FL_params.users_list_2nd = random.sample(selected_users, FL_params.type_count_1st)
    # FL_params.users_list_1st = list(set(selected_users) - set(FL_params.users_list_2nd))
    #
    # # if not FL_params.users_list_2nd:
    # #     print("users_list_2nd is empty.")
    # #     FL_params.multi_model=False
    # #     FL_params.model='cifar'
    # #
    # # elif not FL_params.users_list_1st:
    # #     print("users_list_1st is empty.")
    # #     FL_params.multi_model = False
    # #     FL_params.model = 'mnist'
    # # else:
    # #     print("Neither list1 nor list2 is empty.")
    #
    #
    # # FL_params.users_list_2nd = [11, 12, 13, 14, 15]
    # # FL_params.users_list_1st = [1, 2, 3, 4, 5]
    # print(f'type_count_1st: {FL_params.type_count_1st}, type_count_2nd: {FL_params.type_count_2nd}')
    #
    # old_client_models = {'local_1': [], 'global_1': [], 'gate_1': [], 'local_2': [], 'global_2': [], 'gate_2': []}
    #
    # # init_global_model = model_init(FL_params.data_name, device)
    # # client_all_loaders, test_loader = dataloader_init(FL_params)
    # # vocab = None
    #
    # opt_out = []
    # opt_out_set = set(FL_params.opt_out)
    # opt_in_1st = [user for user in FL_params.users_list_1st if user not in opt_out_set]
    # opt_in_2nd = [user for user in FL_params.users_list_2nd if user not in opt_out_set]
    # FL_params.type_count_1st = len(opt_in_1st)
    # FL_params.type_count_2nd = len(opt_in_2nd)
    # print(f'type_count_1st: {FL_params.type_count_1st}, type_count_2nd: {FL_params.type_count_2nd}')

    # # opt_out.append(FL_params.opt_out)
    # opt_in = []
    # opt_in.append(opt_in_1st)
    # opt_in.append(opt_in_2nd)
    # FL_params.opt_in = opt_in
    # print("opt in", opt_in)
    # counter = 0
    #
    # print('#' * 7, 'Loading dataset for federated learning', '#' * 7)
    # dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test = splitExpertData(FL_params)
    #
    # class_dict = {
    #     'first': list(set([label for _, label in dataset_train[0]])),
    #     'second': list(set([label for _, label in dataset_train[1]])) #if FL_params.data_1 != FL_params.data_2 else []
    # }
    #
    # FL_params.data_class = class_dict
    #
    # class_len_dict = {
    #     'first': len(class_dict['first']),
    #     'second': len(class_dict['second']) if class_dict['second'] else 0
    # }
    #
    # FL_params.all_classes = class_len_dict
    #
    # # 假设 splitExpertModel 函数返回的是一个包含多个模型的字典
    # client_models = splittExpertModel(FL_params)
    #
    # # 初始化模型
    # net_glob_fedAvg = [client_models['global_1'], client_models['global_2']]
    # gate_model = [client_models['gate_1'], client_models['gate_2']]
    # net_locals = [client_models['local_1'], client_models['local_2']]
    #
    # # 创建 DataLoader
    # first_client_loaders = create_user_dataloaders(dataset_train[0], dict_users[0],
    #                                                batch_size=FL_params.local_batch_size)
    # first_client_val_loaders = create_user_dataloaders(dataset_train[0], dict_users_val[0],
    #                                                    batch_size=FL_params.local_batch_size)
    # first_client_test_loaders = create_user_dataloaders(dataset_test[0], dict_users_test[0],
    #                                                     batch_size=FL_params.local_batch_size)
    #
    # if FL_params.model_1 != FL_params.model_2:
    #     second_client_loaders = create_user_dataloaders(dataset_train[1], dict_users[1],
    #                                                     batch_size=FL_params.local_batch_size)
    #     second_client_val_loaders = create_user_dataloaders(dataset_train[1], dict_users_val[1],
    #                                                         batch_size=FL_params.local_batch_size)
    #     second_client_test_loaders = create_user_dataloaders(dataset_test[1], dict_users_test[1],
    #                                                          batch_size=FL_params.local_batch_size)
    # else:
    #     # 如果两个数据集相同，复用第一个数据集的加载器
    #     second_client_loaders = first_client_loaders
    #     second_client_val_loaders = first_client_val_loaders
    #     second_client_test_loaders = first_client_test_loaders
    #
    # # 初始化测试加载器
    # test_loader = []
    # test_loader.extend(first_client_test_loaders)
    # if FL_params.model_1 != FL_params.model_2:
    #     test_loader.extend(second_client_test_loaders)
    # else:
    #     test_loader.extend(first_client_test_loaders)  # 如果数据集相同，直接复用
    #
    # # opt_out = []
    # # opt_out_set = set(FL_params.opt_out)
    # # # 从 users_list_1st 和 users_list_2nd 中移除 opt_out 中的元素
    # # opt_in_1st = [user for user in FL_params.users_list_1st if user not in opt_out_set]
    # # opt_in_2nd = [user for user in FL_params.users_list_2nd if user not in opt_out_set]
    # # FL_params.type_count_1st=len(opt_in_1st)
    # # FL_params.type_count_2nd=len(opt_in_2nd)
    # # print(f'type_count_1st: {FL_params.type_count_1st}, type_count_2nd: {FL_params.type_count_2nd}')
    # #
    # # opt_out.append(FL_params.opt_out)
    # # opt_in = []
    # # opt_in.append(opt_in_1st)
    # # opt_in.append(opt_in_2nd)
    # # FL_params.opt_in = opt_in
    # # print("opt in", opt_in)
    # # counter = 0
    #
    # # 生成新的嵌套列表
    # new_list = []
    # for sublist in opt_in:
    #     new_sublist = []
    #     for _ in sublist:
    #         new_sublist.append(counter)
    #         counter += 1
    #     new_list.append(new_sublist)
    #
    # idxs_users1 = list(range(len(opt_in[0])))
    # idxs_users2 = list(range(len(opt_in[1])))

    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadowl_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    net_mix_glob_fedAvg = []
    if FL_params.type_count_2nd:
        # model_type1=FL_params.data_name#='mnist'#'cifar'
        # std_time_fedavg = time.time()
        # net_glob_fedAvg1, writer = mix_Train(FL_params.data_name, opt_in[0], dataset_train[0], dataset_test[0], dict_users[0],
        #                                      dict_users_val[0],
        #                                      dict_users_test[0],
        #                                      net_glob_fedAvg[0],
        #                                      writer, FL_params, vocab, first_client_loaders, first_client_val_loaders)
        # print('type one:',type(net_glob_fedAvg1))
        # end_time_fedavg = time.time()
        # time_learn_fedavg = (std_time_fedavg - end_time_fedavg)
        # print(" Learning 1 time consuming = {} secnods".format(-time_learn_fedavg))
        # net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        # torch.save(net_glob_fedAvg1.state_dict(), '1500cifar_cnn_global.pth')
        # exit(0)

        if FL_params.model_1 == 'resnet_cifar':
            print('-------Using pre-trained model for resnet cifar10------')
            net_glob_fedAvg1 = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)

            gm_checkpoint1 = torch.load('1500multi_cifar_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        elif FL_params.model_1 == 'cnn_cifar':
            print('-------Using pre-trained model for cnn cifar------')
            net_glob_fedAvg1 = CNNCifar(FL_params).to(FL_params.device)
            # net_glob_fedAvg1 = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
            gm_checkpoint1 = torch.load('1500cifar_cnn_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        elif FL_params.model_1 == 'purchase':
            net_glob_fedAvg1 = Net_purchase().to(FL_params.device)
            gm_checkpoint1 = torch.load('1500_purchase_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        elif FL_params.model_1 == 'adult':
            net_glob_fedAvg1 = Net_adult().to(FL_params.device)
            gm_checkpoint1 = torch.load('1500_adult_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        elif FL_params.model_1 == 'cnn_mnist':
            print('-------Using pre-trained model for mnist------')
            net_glob_fedAvg1 = CNNFashion(FL_params).to(FL_params.device)
            # net_glob_fedAvg2 = Net_adult().to(FL_params.device)
            gm_checkpoint1 = torch.load('1500multi_mnist_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)
        elif FL_params.model_1 == 'cnn_fashion':
            print('-------Using pre-trained model for mnist------')
            net_glob_fedAvg1 = CNNFashion(FL_params).to(FL_params.device)
            # fashion-mnist
            gm_checkpoint1 = torch.load('1500fashionmulti_mnist_global.pth', map_location='cpu')
            net_glob_fedAvg1.load_state_dict(gm_checkpoint1, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg1)

        finetune_time = 0

        std_time_finetune = time.time()
        print('fintune train clietn loader: ', len(first_client_loaders))
        finetuned1, locals_nets1, val_acc_ft1, train_acc_ft1, val_acc_locals1, train_acc_locals1, client1, ft_time1 = train_clients(
            writer, FL_params.data_1,
            idxs_users1, dataset_train[0], dataset_test[0], dict_users[0], dict_users_val[0], dict_users_test[0],
            net_mix_glob_fedAvg[0], net_locals[0], FL_params, first_client_loaders, first_client_val_loaders)

        save_model(finetuned1[0], "ckpt/finetuned", f"finetuned1_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # filename = f"finetuned1_{timestamp}.pth"
        # save_dir = "ckpt/finetuned"
        #
        # # 检查目录是否存在，如果不存在则创建它
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # torch.save(finetuned1[0].state_dict(), f"{save_dir}/{filename}")

        end_time_finetune = time.time()
        time_learn_finetune = (end_time_finetune - std_time_finetune)
        finetune_time = finetune_time + time_learn_finetune
        print(" Fine-tuning 1 time consuming = {} seconds".format(time_learn_finetune))

        old_client_models['local_1'] = copy.deepcopy(finetuned1)
        old_client_models['global_1'] = copy.deepcopy(finetuned1[0])
        # t=[]
        for _ in range(FL_params.type_count_2nd):
            t = []
            gate = copy.deepcopy(client_models['gate_1'])
            t.append(gate)
            old_client_models['gate_1'].extend(t)

    if FL_params.type_count_1st:
        # model_type2=FL_params.data2_name#='mnist'
        # std_time_fedavg2 = time.time()
        # net_glob_fedAvg2, writer = mix_Train(FL_params.data2_name, opt_in[1], dataset_train[1], dataset_test[1], dict_users[1],
        #                                      dict_users_val[1],
        #                                      dict_users_test[1],
        #                                      net_glob_fedAvg[1],
        #                                      writer, FL_params, vocab, second_client_loaders, second_client_val_loaders)
        # print('type two:', type(net_glob_fedAvg2))
        # end_time_fedavg2 = time.time()
        # time_learn_fedavg2 = (std_time_fedavg2 - end_time_fedavg2)
        # torch.save(net_glob_fedAvg2.state_dict(), '1500_adult_global.pth')

        # net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        # print(" Learning 2 time consuming = {} secnods".format(-time_learn_fedavg2))
        if FL_params.model_2 == 'cnn_mnist':
            print('-------Using pre-trained model for mnist------')
            net_glob_fedAvg2 = CNNFashion(FL_params).to(FL_params.device)
            # net_glob_fedAvg2 = Net_adult().to(FL_params.device)
            gm_checkpoint2 = torch.load('1500multi_mnist_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        elif FL_params.model_2 == 'cnn_fashion':
            print('-------Using pre-trained model for mnist------')
            net_glob_fedAvg2 = CNNFashion(FL_params).to(FL_params.device)
            gm_checkpoint2 = torch.load('1500fashionmulti_mnist_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        elif FL_params.model_2 == 'adult':
            net_glob_fedAvg2 = Net_adult().to(FL_params.device)
            gm_checkpoint2 = torch.load('1500_adult_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        elif FL_params.model_2 == 'cnn_cifar':
            print('-------Using pre-trained model for cnn cifar------')
            net_glob_fedAvg2 = CNNCifar(FL_params).to(FL_params.device)
            gm_checkpoint2 = torch.load('1500cifar_cnn_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        elif FL_params.model_2 == 'purchase':
            net_glob_fedAvg2 = Net_purchase().to(FL_params.device)
            gm_checkpoint2 = torch.load('1500_purchase_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)
        elif FL_params.model_2 == 'resnet_cifar':
            net_glob_fedAvg2 = models.resnet18(pretrained=False, num_classes=10).to(FL_params.device)
            gm_checkpoint2 = torch.load('1500multi_cifar_global.pth', map_location='cpu')
            net_glob_fedAvg2.load_state_dict(gm_checkpoint2, strict=False)
            net_mix_glob_fedAvg.append(net_glob_fedAvg2)

        std_time_finetune2 = time.time()
        print("Check mode model2", type(net_glob_fedAvg2))
        finetuned2, locals_nets2, val_acc_ft2, train_acc_ft2, val_acc_locals2, train_acc_locals2, client2, ft_time2 \
            = train_clients(writer, FL_params.data_2, idxs_users2, dataset_train[1], dataset_test[1],
                            dict_users[1], dict_users_val[1], dict_users_test[1], net_glob_fedAvg2,
                            net_locals[1], FL_params, second_client_loaders, second_client_val_loaders)

        end_time_finetune2 = time.time()
        time_learn_finetune2 = (end_time_finetune2 - std_time_finetune2)
        finetune_time = finetune_time + time_learn_finetune2

        print(" Fine-tuning 1 time consuming = {} seconds".format(time_learn_finetune2))
        save_model(finetuned2[0], "ckpt/finetuned", f"finetuned2_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")

        old_client_models['local_2'] = copy.deepcopy(finetuned2)
        old_client_models['global_2'] = copy.deepcopy(finetuned2[0])
        for _ in range(FL_params.type_count_1st):
            t = []
            gate = copy.deepcopy(client_models['gate_2'])
            t.append(gate)
            old_client_models['gate_2'].extend(t)

        # experts = [old_client_models['local_1'], old_client_models['global_1'], old_client_models['gate_1'],
        #            old_client_models['local_2'], old_client_models['global_2'], old_client_models['gate_2']]
        # print('Old client models', len(old_client_models['local_1']), len(old_client_models['local_2']),
        #       len(old_client_models['gate_1']), len(old_client_models['gate_2']))

        # client_models = []
        # client_models.extend(locals_nets1)
        # client_models.extend(locals_nets2)

        idxs_users = []
        idxs_users.append(opt_in[0])
        idxs_users.append(opt_in[1])
        finetune_list = []
        finetune_list.extend(finetuned1)
        finetune_list.extend(finetuned2)

        net_globals_list = []
        for _ in range(FL_params.type_count_2nd):
            net_globals_list.extend([net_glob_fedAvg1])
        for _ in range(FL_params.type_count_1st):
            net_globals_list.extend([net_glob_fedAvg2])

        print('#' * 7, ' Global Gating Start', '#' * 7)

        std_time_mix = time.time()
        rest_dict_users = []
        unlearn_dict_users = []

        (tfidf_scores, updated_dict_users, unlearn_dict_users, updated_test_dict_users, unlearn_test_dict_users,
         updated_dict_users1, unlearn_dict_users1, updated_test_dict_users1, unlearn_test_dict_users1,
         user_dataloaders1, unlearn_dataloaders1, user_dataloaders1_test, unlearn_dataloaders1_test,
         user_dataloaders2, unlearn_dataloaders2, user_dataloaders2_test, unlearn_dataloaders2_test) = (
            mix_train_clients(idxs_users1, idxs_users2, dict_users, dict_users_test, dataset_train, dataset_test,
                                client1, client2, net_globals_list, gate_model, finetune_list, client_models, FL_params))

        end_time_mix = time.time()
        time_learn_mix = (end_time_mix - std_time_mix)
        gate_time = 0
        gate_time = gate_time + time_learn_mix
        print(" Global gating(unlearning) time consuming = {} seconds".format(time_learn_mix))
        print('#' * 7, ' Global Gating End', '#' * 7)

        filtered_opt_in = FL_params.opt_in

        FL_params.seperate_mix = True
        val_acc_e2e1 = 1
        val_acc_fedavg1 = 1
        mix_time = 0

        if filtered_opt_in[0]:
            std_time_local_mix = time.time()
            if updated_dict_users is None:
                print('Not unlearn class in type one dataset')
                updated_dict_users = dict_users[0]
                updated_test_dict_users = dict_users_test[0]
            if user_dataloaders1_test is None:
                user_dataloaders1_test = first_client_test_loaders
            mix_local1, mix_global1, mix_gate1, val_acc_e2e1, val_acc_fedavg1, mixed_client_models1 = mix_train_clients_step2(
                writer,
                tfidf_scores[0], FL_params.data_1, filtered_opt_in[0],
                dataset_train[0], dataset_test[0],
                updated_dict_users,
                updated_dict_users, updated_test_dict_users, net_glob_fedAvg1,
                client_models['gate_1'], finetuned1, client_models['local_1'], FL_params, user_dataloaders1, user_dataloaders1)
            end_time_local_mix = time.time()
            time_learn_localmix = (end_time_local_mix - std_time_local_mix)
            mix_time = mix_time + time_learn_localmix
            print("#" * 7, "Local gating 1st type time consuming = {} seconds".format(time_learn_localmix), "#" * 7)
            save_model(mix_global1[0], "ckpt/pruned", f"pruned1_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")

            test_idxs = []
            for sublist in filtered_opt_in[0]:
                sublist_idxs = []
                if sublist in new_list:
                    sublist_idxs.extend(new_list.index(sublist))
                # for item in sublist:
                # for i, original_sublist in enumerate(new_list):
                #     if sublist in original_sublist:
                #         sublist_idxs.append(original_sublist.index(sublist))
                #         break
                test_idxs.extend(sublist_idxs)
            localtest_acc_local1, localtest_acc_ft1, localtest_acc_e2e1, localtest_acc_fedavg1, test_acc_local1, test_acc_ft1, test_acc_e2e1, test_acc_fedavg1 = test_clients(
                writer, tfidf_scores[0], FL_params.data_1, idxs_users[0], dataset_train[0],
                updated_dict_users, updated_dict_users, dataset_test[0], user_dataloaders1_test,
                unlearn_dataloaders1_test, unlearn_test_dict_users, updated_test_dict_users,
                net_glob_fedAvg1,
                # mix_local1,
                locals_nets1,
                finetuned1, mix_local1, mix_global1,
                mix_gate1, user_dataloaders1, FL_params)
        # val_acc_e2e2=1
        # val_acc_fedavg2=1
        if filtered_opt_in[1]:
            std_time_local_mix2 = time.time()
            if updated_dict_users1 is None or updated_test_dict_users1 is None:
                print('Not unlearn class in type two dataset')
                updated_dict_users1 = dict_users[1]
                updated_test_dict_users1 = dict_users_test[1]
                user_dataloaders2_test = second_client_test_loaders
            # mix_local1, mix_global1, mix_gate1, val_acc_e2e1, val_acc_fedavg1, mixed_client_models1 = mix_train_clients_step2(
            #     writer,
            #     tfidf_scores[0], FL_params.data_1, filtered_opt_in[0],
            #     dataset_train[0], dataset_test[0],
            #     updated_dict_users,
            #     updated_dict_users, updated_test_dict_users, net_glob_fedAvg1,
            #     client_models['gate_1'], finetuned1, client_models['local_1'], FL_params, vocab, user_dataloaders1,
            #     user_dataloaders1)

            mix_local2, mix_global2, mix_gate2, val_acc_e2e2, val_acc_fedavg2, mixed_client_models2 = mix_train_clients_step2(
                writer,
                tfidf_scores[1], FL_params.data_2, filtered_opt_in[1],
                dataset_train[1], dataset_test[1],
                updated_dict_users1,
                updated_dict_users1, updated_dict_users1,
                net_glob_fedAvg2, client_models['gate_2'], finetuned2, client_models['local_2'], FL_params,
                user_dataloaders2, user_dataloaders2)
            end_time_local_mix2 = time.time()
            time_learn_localmix2 = (end_time_local_mix2 - std_time_local_mix2)
            mix_time = mix_time + time_learn_localmix2
            print("#" * 7, "Local gating 2nd type time consuming = {} seconds".format(time_learn_localmix2), "#" * 7)

            save_model(mix_global2[0], "ckpt/pruned", f"pruned2_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")

            # # Test clients
            test_idxs = []
            for sublist in filtered_opt_in[1]:
                sublist_idxs = []
                if sublist in new_list:
                    sublist_idxs.extend(new_list.index(sublist))

                test_idxs.extend(sublist_idxs)
            localtest_acc_local2, localtest_acc_ft2, localtest_acc_e2e2, localtest_acc_fedavg2, test_acc_local2, test_acc_ft2, test_acc_e2e2, test_acc_fedavg2 = test_clients(
                writer, tfidf_scores[1], FL_params.data_2, idxs_users[1], dataset_train[1],
                updated_dict_users1, updated_dict_users1, dataset_test[1], user_dataloaders2_test,
                unlearn_dataloaders2_test, unlearn_test_dict_users1, updated_test_dict_users1,
                net_glob_fedAvg2, locals_nets2,
                # mix_local2,
                finetuned2, mix_local2, mix_global2,
                mix_gate2, user_dataloaders2, FL_params)
        # Calculate average accuracies
        val_acc_avg_locals1 = sum(val_acc_locals1) / len(val_acc_locals1)
        train_acc_avg_locals1 = sum(train_acc_locals1) / len(train_acc_locals1)
        val_acc_avg_e2e1 = sum(val_acc_e2e1) / len(val_acc_e2e1)
        val_acc_avg_fedavg1 = (sum(val_acc_fedavg1) / len(val_acc_fedavg1))
        ft_val_acc1 = (sum(val_acc_ft1) / len(val_acc_ft1))
        ft_train_acc1 = sum(train_acc_ft1) / len(train_acc_ft1)

        val_acc_avg_locals2 = (sum(val_acc_locals2) / len(val_acc_locals2))
        train_acc_avg_locals2 = sum(train_acc_locals2) / len(train_acc_locals2)
        val_acc_avg_e2e2 = sum(val_acc_e2e2) / len(val_acc_e2e2)
        val_acc_avg_fedavg2 = (sum(val_acc_fedavg2) / len(val_acc_fedavg2))
        ft_val_acc2 = (sum(val_acc_ft2) / len(val_acc_ft2))
        ft_train_acc2 = sum(train_acc_ft2) / len(train_acc_ft2)

        print('#' * 7, 'Retraining Start', '#' * 7)

        retrain_start_time = time.time()

        retrain_test_loss, retrain_test_acc, retrain_pre, retrain_rec, retrain_f1 = [], [], [], [], []
        reval_loss_avg, reval_acc_avg, retrain_loss_avg, retrain_acc_avg = [], [], [], []

        retrained_model = []
        retrain_test_loader = []
        for cls in FL_params.unlearn_class:
            if cls < (FL_params.all_classes['first']):
                print('Retrain first')
                retrain_test_loader.append(first_client_test_loaders)

                retrain_model, retrain_loss_av, retrain_acc_av, reval_loss_av, reval_acc_av = quick_retrain(
                    FL_params.model_1, FL_params.users_list_1st, dataset_train[0], dataset_test[0], dict_users[0],
                    dict_users_val[0],
                    dict_users_test[0], net_glob_fedAvg1, writer, FL_params, first_client_loaders,
                    first_client_val_loaders)
                retrained_model.append(retrain_model)
                retrain_loss_avg.append(retrain_loss_av)
                retrain_acc_avg.append(retrain_acc_av)
                reval_loss_avg.append(reval_loss_av)
                reval_acc_avg.append(reval_acc_av)
                save_model(retrain_model, "ckpt/retrain", f"retrain1_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")

            else:
                print('Retrain second')
                retrain_test_loader.append(second_client_test_loaders)

                retrain_model, retrain_loss_av, retrain_acc_av, reval_loss_av, reval_acc_av = quick_retrain(
                    FL_params.model_2, FL_params.users_list_2nd, dataset_train[1], dataset_test[1], dict_users[1],
                    dict_users_val[1],
                    dict_users_test[1], net_glob_fedAvg2, writer, FL_params, second_client_loaders,
                    second_client_val_loaders)
                retrained_model.append(retrain_model)
                retrain_loss_avg.append(retrain_loss_av)
                retrain_acc_avg.append(retrain_acc_av)
                reval_loss_avg.append(reval_loss_av)
                reval_acc_avg.append(reval_acc_av)
                save_model(retrain_model, "ckpt/retrain", f"retrain2_{FL_params.p}p_unlearn{FL_params.unlearn_class_num}class_optout{FL_params.opt_out}")
        for i in range(len(retrained_model)):
            (test_loss, test_acc, pre, rec, f1_score) = test(retrained_model[i], retrain_test_loader[i][-1])
            retrain_test_loss.append(test_loss)
            retrain_test_acc.append(test_acc)
            retrain_pre.append(pre)
            retrain_rec.append(rec)
            retrain_f1.append(f1_score)

        retrain_end_time = time.time()
        retrain_time = (retrain_end_time - retrain_start_time)
        print('retrain time:', retrain_time)

        print(
            f'Rretrain_loss_avg={retrain_loss_avg}, retrain_acc_avg={retrain_acc_avg}, reval_loss_avg={reval_loss_avg}, reval_acc_avg={reval_acc_avg}')
        print('#' * 7, 'Retraining End', '#' * 7)


        # Evaluation
        if unlearn_dict_users:
            print('Update 1st target loader')
            first_client_unlearn_target_loaders = unlearn_dataloaders1
        if unlearn_dict_users1:
            print('Update 2nd target loader')
            second_client_unlearn_target_loaders = unlearn_dataloaders2

        fedavg_test_loss, fedavg_test_acc, fedavg_test_pre, fedavg_test_rec, fedavg_test_f1=0,0,0,0,0
        federaser_test_loss, federaser_test_acc, federaser_test_pre, federaser_test_rec, federaser_test_f1=0,0,0,0,0
        if unlearn_dataloaders1_test:
            print('TESTING THE FIRST TYPE CLIENTS')
            fedavg_test_loss, fedavg_test_acc, fedavg_test_pre, fedavg_test_rec, fedavg_test_f1 = test(
                old_client_models['global_1'], unlearn_dataloaders1_test)#first_client_test_loaders[0])
            federaser_test_loss, federaser_test_acc, federaser_test_pre, federaser_test_rec, federaser_test_f1 = test(
                mix_global1[-1], unlearn_dataloaders1_test)#first_client_test_loaders[0])
        fedavg_test_loss1, fedavg_test_acc1, fedavg_test_pre1, fedavg_test_rec1, fedavg_test_f11=0,0,0,0,0
        federaser_test_loss1, federaser_test_acc1, federaser_test_pre1, federaser_test_rec1, federaser_test_f11=0,0,0,0,0
        if unlearn_dataloaders2_test:
            print('TESTING THE SECOND TYPE CLIENTS')
            fedavg_test_loss1, fedavg_test_acc1, fedavg_test_pre1, fedavg_test_rec1, fedavg_test_f11 = test(
                old_client_models['global_2'], unlearn_dataloaders2_test)#second_client_test_loaders[0])
            federaser_test_loss1, federaser_test_acc1, federaser_test_pre1, federaser_test_rec1, federaser_test_f11 = test(
            mix_global2[-1], unlearn_dataloaders2_test)#second_client_test_loaders[0])

        fedavg_target_loss = []
        fedavg_target_acc = []
        fedavg_target_pre = []
        fedavg_target_rec = []
        fedavg_target_f1 = []

        federaser_target_loss = []
        federaser_target_acc = []
        federaser_target_pre = []
        federaser_target_rec = []
        federaser_target_f1 = []

        fedretrain_target_loss = []
        fedretrain_target_acc = []
        fedretrain_target_pre = []
        fedretrain_target_rec = []
        fedretrain_target_f1 = []

        target_loader = []
        old_target_loss, old_target_acc, old_target_pre, old_target_rec, old_target_f1 = [], [], [], [], []
        unlearned_target_loss, unlearned_target_acc, unlearned_target_pre, unlearned_target_rec, unlearned_target_f1 = [], [], [], [], []
        retrain_target_loss, retrain_target_acc, retrain_target_pre, retrain_target_rec, retrain_target_f1 = [], [], [], [], []
        for cls in FL_params.unlearn_class:
            if cls < 10:
                print('1st')
                target_loader.append(first_client_unlearn_target_loaders)
                loss, acc, pre, rec, f1 = test(client_models['global_1'], target_loader[-1])
                fedavg_target_loss.append(loss)
                fedavg_target_acc.append(acc)
                fedavg_target_pre.append(pre)
                fedavg_target_rec.append(rec)
                fedavg_target_f1.append(f1)

                loss, acc, pre, rec, f1 = test(mix_global1[-1], target_loader[-1])
                federaser_target_loss.append(loss)
                federaser_target_acc.append(acc)
                federaser_target_pre.append(pre)
                federaser_target_rec.append(rec)
                federaser_target_f1.append(f1)

                if isinstance(retrained_model[0], type(mix_global1[-1])):
                    loss, acc, pre, rec, f1 = test(retrained_model[0], target_loader[-1])
                    fedretrain_target_loss.append(loss)
                    fedretrain_target_acc.append(acc)
                    fedretrain_target_pre.append(pre)
                    fedretrain_target_rec.append(rec)
                    fedretrain_target_f1.append(f1)

            else:
                print('2nd')
                target_loader.append(second_client_unlearn_target_loaders)
                # print(type(second_client_unlearn_target_loaders),type(second_client_unlearn_target_loaders[0]))
                loss, acc, pre, rec, f1 = test(old_client_models['global_2'], second_client_unlearn_target_loaders[0])
                fedavg_target_loss.append(loss)
                fedavg_target_acc.append(acc)
                fedavg_target_pre.append(pre)
                fedavg_target_rec.append(rec)
                fedavg_target_f1.append(f1)

                loss, acc, pre, rec, f1 = test(mix_global2[-1], target_loader[-1])
                federaser_target_loss.append(loss)
                federaser_target_acc.append(acc)
                federaser_target_pre.append(pre)
                federaser_target_rec.append(rec)
                federaser_target_f1.append(f1)

                if isinstance(retrained_model[-1],type(mix_global2[-1])):
                    loss, acc, pre, rec, f1 = test(retrained_model[-1], target_loader[-1])
                    fedretrain_target_loss.append(loss)
                    fedretrain_target_acc.append(acc)
                    fedretrain_target_pre.append(pre)
                    fedretrain_target_rec.append(rec)
                    fedretrain_target_f1.append(f1)

        print(5 * "*" + "   Result Summary  " + 5 * "*")
        # if unlearn_dataloaders1_test:
        print(f"[FedEraser] Test set: Average loss = {federaser_test_loss}, Average acc = {federaser_test_acc}")
        print(f"[FedAvg] Test set: Average loss = {fedavg_test_loss}, Average acc = {fedavg_test_acc}")
        print('fedavg_test_loss, fedavg_test_acc, fedavg_test_pre, fedavg_test_rec,fedavg_test_f1', fedavg_test_loss,
              fedavg_test_acc, fedavg_test_pre, fedavg_test_rec, fedavg_test_f1)
        print('federaser_test_loss, federaser_test_acc,federaser_test_pre,federaser_test_rec,federaser_test_f1',
              federaser_test_loss, federaser_test_acc, federaser_test_pre, federaser_test_rec, federaser_test_f1)
        print("\n")
        print(
            f"[FedEraser] Target set: Average loss = {federaser_target_loss[-1]}, Average acc ={federaser_target_acc} ")
        print(f"[FedAvg] Target set: Average loss = {fedavg_target_loss[-1]}, Average acc = {fedavg_target_acc}")

        # if unlearn_dataloaders2_test:
        print(f"11[FedEraser] Test set: Average loss = {federaser_test_loss1}, Average acc = {federaser_test_acc1}")
        print(f"11[FedAvg] Test set: Average loss = {fedavg_test_loss1}, Average acc = {fedavg_test_acc1}")
        print('fedavg_test_loss, fedavg_test_acc, fedavg_test_pre, fedavg_test_rec,fedavg_test_f1',
              fedavg_test_loss1,
              fedavg_test_acc1, fedavg_test_pre1, fedavg_test_rec1, fedavg_test_f11)

        print('11fedavg_test_loss, fedavg_test_acc, fedavg_test_pre, fedavg_test_rec,fedavg_test_f1', fedavg_test_loss1,
              fedavg_test_acc1, fedavg_test_pre1, fedavg_test_rec1, fedavg_test_f11)
        print('11federaser_test_loss, federaser_test_acc,federaser_test_pre,federaser_test_rec,federaser_test_f1',
              federaser_test_loss1, federaser_test_acc1, federaser_test_pre1, federaser_test_rec1, federaser_test_f11)
        print("\n")
        print(
            f"[FedEraser] Target set: Average loss = {federaser_target_loss[-1]}, Average acc ={federaser_target_acc} ")
        print(f"[FedAvg] Target set: Average loss = {fedavg_target_loss[-1]}, Average acc = {fedavg_target_acc}")

        print(f"[FedRetrain] Test set: Average loss = {retrain_test_loss}, Average acc = {retrain_test_acc}")
        print(
            f"[FedRetrain] Target set: Average loss = {fedretrain_target_loss}, Average acc = {fedretrain_target_acc}")

        print(60 * '=')
        print("Step5. Membership Inference Attack aganist GM...")
        ACC_old = 0
        PRE_old = 0
        ACC_retrain = 0
        PRE_retrain = 0
        ACC_unlearn = 0
        PRE_unlearn = 0
        ACC_old2 = 0
        PRE_old2 = 0
        ACC_retrain2 = 0
        PRE_retrain2 = 0
        ACC_unlearn2 = 0
        PRE_unlearn2 = 0

        if unlearn_dataloaders1_test or unlearn_dataloaders2_test:
            if isinstance(mix_local1[-1], type(client_models['global_1'])):
                (ACC_old, PRE_old, rec_old, f1_old, ACC_retrain, PRE_retrain, ACC_unlearn, PRE_unlearn, rec_unlearn,
                 f1_unlearn) = membership_inference_attack(
                    filtered_opt_in[0], client_models['global_1'],
                    retrained_model[0], mix_local1, user_dataloaders1,
                    user_dataloaders1_test, 'first', FL_params)
                print(f"ACC_old1: {ACC_old}, PRE_old: {PRE_old}, rec_old: {rec_old}, f1_old: {f1_old}")
                print(f"ACC_unlearn: {ACC_unlearn}, PRE_unlearn: {PRE_unlearn},rec_unlearn: {rec_unlearn}, f1_unlearn: {f1_unlearn}")

            if FL_params.if_retrain:
                print(f"ACC_retrain: {ACC_retrain}, PRE_retrain: {PRE_retrain}")
        # if unlearn_dict_users1:
            if isinstance(mix_local2[-1], type(client_models['global_2'])):
                ACC_old2, PRE_old2, rec_old2, f1_old2, ACC_retrain2, PRE_retrain2, ACC_unlearn2, PRE_unlearn2, rec_unlearn2, f1_unlearn2 = membership_inference_attack(
                    filtered_opt_in[1],
                    client_models['global_2'], retrained_model[-1], mix_local2, user_dataloaders2,
                    user_dataloaders2_test,'second', FL_params)

                # Print results
                print(f"ACC_old2: {ACC_old2}, PRE_old: {PRE_old2}, rec_old: {rec_old2}, f1_old: {f1_old2}")
                print(f"ACC_unlearn: {ACC_unlearn2}, PRE_unlearn: {PRE_unlearn2},rec_unlearn: {rec_unlearn2}, f1_unlearn: {f1_unlearn2}")

            if FL_params.if_retrain:
                print(f"ACC_retrain: {ACC_retrain2}, PRE_retrain: {PRE_retrain2}")

        FL_params.seperate_mix = False
        unlearn_time = mix_time + gate_time

        # Save results
        csv_file = '0912mode.csv'

        # 定义表头和数据字典
        header = ['opt_out', 'unlearn_tag', 'finetune_time', 'unlearn_time', 'retrain_time', 'unlearn_class_num',
                  'unlearn_class', 'data_1', 'data_2', 'model_1', 'model_2', 'global_epoch', 'local_epoch', 'N_client',
                  'p', 'local_lr',
                  'val_acc_avg_e2e1', 'val_acc_avg_locals1', 'val_acc_avg_fedavg1', 'ft_val_acc1', 'ft_train_acc1',
                  'train_acc_avg_locals1', 'localtest_acc_fedavg1', 'localtest_acc_local1', 'localtest_acc_ft1',
                  'localtest_acc_e2e1', 'test_acc_fedavg1', 'test_acc_local1', 'test_acc_ft1', 'test_acc_e2e1',

                  'val_acc_avg_e2e2', 'val_acc_avg_locals2', 'val_acc_avg_fedavg2', 'ft_val_acc2', 'ft_train_acc2',
                  'train_acc_avg_locals2', 'localtest_acc_fedavg2', 'localtest_acc_local2', 'localtest_acc_ft2',
                  'localtest_acc_e2e2', 'test_acc_fedavg2', 'test_acc_local2', 'test_acc_ft2', 'test_acc_e2e2',

                  'retrain_loss_avg','retrain_acc_avg','reval_loss_avg','reval_acc_avg',

                  'ACC_old', 'PRE_old', 'ACC_retrain', 'PRE_retrain', 'ACC_unlearn', 'PRE_unlearn',
                  'ACC_old2', 'PRE_old2', 'ACC_retrain2', 'PRE_retrain2', 'ACC_unlearn2', 'PRE_unlearn2',

                  'fedavg_test_loss',
                  'fedavg_test_acc',
                  'fedavg_test_pre',
                  'fedavg_test_rec',
                  'fedavg_test_f1',
                  'fedavg_test_loss1', 'fedavg_test_acc1', 'fedavg_test_pre1', 'fedavg_test_rec1', 'fedavg_test_f11',

                  'federaser_test_loss',
                  'federaser_test_acc',
                  'federaser_test_pre',
                  'federaser_test_rec',
                  'federaser_test_f1',
                  'federaser_test_loss1', 'federaser_test_acc1', 'federaser_test_pre1', 'federaser_test_rec1',
                  'federaser_test_f11',

                  'fedavg_target_loss',
                  'fedavg_target_acc',
                  'fedavg_target_pre',
                  'fedavg_target_rec',
                  'fedavg_targetf1',

                  'federaser_target_loss',
                  'federaser_target_acc',
                  'federaser_target_pre',
                  'federaser_target_rec',
                  'federaser_target_f1',

                  'retrain_test_loss',
                  'retrain_test_acc',
                  'retrain_pre',
                  'retrain_rec',
                  'retrain_f1',
                  'retrain_time',

                  'fedretrain_target_loss',
                  'fedretrain_target_acc',
                  'fedretrain_target_pre',
                  'fedretrain_target_rec',
                  'fedretrain_target_f1'

                  ]


        # 创建数据字典
        data = {
            'opt_out':FL_params.opt_out,
            'unlearn_tag': FL_params.unlearn_class_tag,
            'finetune_time': finetune_time,
            'unlearn_time': unlearn_time,
            'unlearn_class_num': FL_params.unlearn_class_num,
            'unlearn_class': FL_params.unlearn_class,
            'data_1': FL_params.data_1,
            'data_2': FL_params.data_2,
            'model_1': FL_params.model_1,
            'model_2': FL_params.model_2,
            'global_epoch': FL_params.global_epoch,
            'local_epoch': FL_params.local_epoch,
            'N_client': FL_params.N_client,

            'p': FL_params.p,

            'local_lr': FL_params.local_lr,
            'val_acc_avg_e2e1': val_acc_avg_e2e1,
            'val_acc_avg_locals1': val_acc_avg_locals1,
            'val_acc_avg_fedavg1': val_acc_avg_fedavg1,
            'ft_val_acc1': ft_val_acc1,
            'ft_train_acc1': ft_train_acc1,
            'train_acc_avg_locals1': train_acc_avg_locals1,
            'localtest_acc_fedavg1': localtest_acc_fedavg1,
            'localtest_acc_local1': localtest_acc_local1,
            'localtest_acc_ft1': localtest_acc_ft1,
            'localtest_acc_e2e1': localtest_acc_e2e1,
            'test_acc_fedavg1': test_acc_fedavg1,
            'test_acc_local1': test_acc_local1,
            'test_acc_ft1': test_acc_ft1,
            'test_acc_e2e1': test_acc_e2e1,
            'val_acc_avg_e2e2': val_acc_avg_e2e2,
            'val_acc_avg_locals2': val_acc_avg_locals2,
            'val_acc_avg_fedavg2': val_acc_avg_fedavg2,
            'ft_val_acc2': ft_val_acc2,
            'ft_train_acc2': ft_train_acc2,
            'train_acc_avg_locals2': train_acc_avg_locals2,
            'localtest_acc_fedavg2': localtest_acc_fedavg2,
            'localtest_acc_local2': localtest_acc_local2,
            'localtest_acc_ft2': localtest_acc_ft2,
            'localtest_acc_e2e2': localtest_acc_e2e2,
            'test_acc_fedavg2': test_acc_fedavg2,
            'test_acc_local2': test_acc_local2,
            'test_acc_ft2': test_acc_ft2,
            'test_acc_e2e2': test_acc_e2e2,
            'retrain_loss_avg': retrain_loss_avg,
            'retrain_acc_avg': retrain_acc_avg,
            'reval_loss_avg': reval_loss_avg,
            'reval_acc_avg': reval_acc_avg,

            'ACC_old': ACC_old,
            'PRE_old': PRE_old,
            'ACC_retrain': ACC_retrain,
            'PRE_retrain': PRE_retrain,
            'ACC_unlearn': ACC_unlearn,
            'PRE_unlearn': PRE_unlearn,
            'ACC_old2': ACC_old2,
            'PRE_old2': PRE_old2,
            'ACC_retrain2': ACC_retrain2,
            'PRE_retrain2': PRE_retrain2,
            'ACC_unlearn2': ACC_unlearn2,
            'PRE_unlearn2': PRE_unlearn2,

            'fedavg_test_loss': fedavg_test_loss,
            'fedavg_test_acc': fedavg_test_acc,
            'fedavg_test_pre': fedavg_test_pre,
            'fedavg_test_rec': fedavg_test_rec,
            'fedavg_test_f1': fedavg_test_f1,
            'fedavg_test_loss1': fedavg_test_loss1, 'fedavg_test_acc1': fedavg_test_acc1,
            'fedavg_test_pre1': fedavg_test_pre1, 'fedavg_test_rec1': fedavg_test_rec1,
            'fedavg_test_f11': fedavg_test_f11,

            'federaser_test_loss': federaser_test_loss,
            'federaser_test_acc': federaser_test_acc,
            'federaser_test_pre': federaser_target_pre,
            'federaser_test_rec': federaser_test_rec,
            'federaser_test_f1': federaser_test_f1,
            'federaser_test_loss1': federaser_test_loss1, 'federaser_test_acc1': federaser_test_acc1,
            'federaser_test_pre1': federaser_test_pre1, 'federaser_test_rec1': federaser_test_rec1,
            'federaser_test_f11': federaser_test_f11,

            'fedavg_target_loss': fedavg_target_loss,
            'fedavg_target_acc': fedavg_target_acc,
            'fedavg_target_pre': fedavg_target_pre,
            'fedavg_target_rec': fedavg_target_rec,
            'fedavg_targetf1': fedavg_target_f1,

            'federaser_target_loss': federaser_target_loss,
            'federaser_target_acc': federaser_target_acc,
            'federaser_target_pre': federaser_target_pre,
            'federaser_target_rec': federaser_target_rec,
            'federaser_target_f1': federaser_target_f1,

            'retrain_test_loss': retrain_test_loss,
            'retrain_test_acc': retrain_test_acc,
            'retrain_pre': retrain_pre,
            'retrain_rec': retrain_rec,
            'retrain_f1': retrain_f1,
            'retrain_time': retrain_time,

            'fedretrain_target_loss': fedretrain_target_loss,
            'fedretrain_target_acc': fedretrain_target_acc,
            'fedretrain_target_pre': fedretrain_target_pre,
            'fedretrain_target_rec': fedretrain_target_rec,
            'fedretrain_target_f1': fedretrain_target_f1
        }

        # 检查文件是否存在，如果不存在则创建文件并写入表头
        file_exists = os.path.isfile(csv_file)

        # 写入文件
        with open(csv_file, mode='a', newline='') as f1:
            writer = csv.DictWriter(f1, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('fork')
    Federated_Unlearning()
