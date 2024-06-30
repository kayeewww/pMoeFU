# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
# %%
import torch
import numpy as np
import time
from datetime import datetime
import argparse
import os
import copy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset

# ourself libs
# from model_initiation import model_init
from data_preprocess import data_set, dataloader_init, model_init, Net_cifar10, Net_mnist, splitExpertData, \
    splittExpertModel
from Fed_Unlearn_base import federated_learning_unlearning
from membership_inference import train_attack_model, attack
from FL_base import test, FL_Retrain, mix_Train
import warnings
from torch.utils.tensorboard import SummaryWriter
from sample_data import mnist_noniid2, cifar_noniid2
from FederatedAveraging import FedAvg
from ClientUpdate import ClientUpdate
from test_model import test_img, test_img_mix
from class_pruner import Class_pruner
from moe import MoE

from mix_fl import train_clients, test_clients, mix_train_clients, save_results, unlearning_step_once, update_clients, membership_inference_attack
from sample_data import create_user_dataloaders
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10  ## Total number of clients N.
        self.data_name = 'cifar10'
        self.model = 'cnn'
        # self.mix_experts_modelset = ['mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist']
        self.mix_experts = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pretrained = False
        self.pretrained_gms_file = 'mix_global.pth'  # 'ssh20_cifar_global.pth'#'shake20_global.pth'
        self.pretrained_cms_file = 'mix_client.pth'  # 'ssh20_cifar_client.pth'#'shake20_client.pth'
        # self.pretrained_gms_file = 'shake20_global.pth'
        # self.pretrained_cms_file = 'shake20_client.pth'
        self.save_pretrained = False  # False

        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 1  # If want to forget, change None to the client index
        self.forget_clients_num = 1  # 1-10
        self.selected_clients=[]

        self.global_epoch = 20  # 600#20#600  # T
        self.local_epoch = 5  # 10#2#10 # E
        self.finetune_epoch = 500
        self.mix_epoch = 500

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        self.save_acc = 80
        self.save_re_acc = 75

        # If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = False  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5
        self.client_fraction = 10  # Fraction of clients selected per round.

        self.fats_method = 'client'
        self.rouc = 0.3  # 0.2
        # self.rous = 0.1
        self.k_u = -1
        self.unlearn_client = -1
        self.sparsity = 0.05
        self.class_flag = False
        self.rest_data_loader = None  # torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        self.rest_testdata = None
        self.K = 2
        self.b = 1
        self.N_datapoint = 20  # 80 #16
        self.M = 2000  # 100  # 600
        self.if_sample_unlearning = False
        self.selected_K_group = []
        self.tfidf_threshold=0.1
        self.tv_stability_threshold = 2  # 0.01

        self.p = 0.3
        self.opt = 0.5
        self.n_data = 100
        self.n_data_test = 200
        self.n_data_val = 200
        self.overlap = True
        self.frac = 0.1
        self.train_gate_only = False
        self.iid = False
        self.freeze = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Federated Unlearning Arguments')
        parser.add_argument('--forget_clients_num', type=int, default=self.forget_clients_num,
                            help='Number of clients to forget')
        parser.add_argument('--global_epoch', type=int, default=self.global_epoch, help='Number of global epochs')
        parser.add_argument('--local_epoch', type=int, default=self.local_epoch, help='Number of local epochs')
        parser.add_argument('--finetune_epoch', type=int, default=self.finetune_epoch, help='Number of finetune epochs')
        parser.add_argument('--mix_epoch', type=int, default=self.mix_epoch, help='Number of mix epochs')
        parser.add_argument('--data_name', type=str, default='cifar10', help="name of dataset")

        parser.add_argument('--rouc', type=float, default=self.rouc, help='RouC value')
        parser.add_argument('--K', type=int, default=self.K, help='K value')
        parser.add_argument('--M', type=int, default=self.M, help='M value')
        parser.add_argument('--opt', type=float, default=0.5, help='fraction of clients that opt-in (default: 0.5)')
        parser.add_argument('--p', type=float, default=0.3, help='majority class percentage (default: 0.3)')
        parser.add_argument('--n_data', type=float, default=100, help="datasize on each client")
        parser.add_argument('--n_data_test', type=float, default=200, help="test datasize on each client")
        parser.add_argument('--n_data_val', type=float, default=200, help="validation datasize on each client")
        parser.add_argument('--overlap', action='store_true',
                            help='whether to allow label overlap between clients or not')
        parser.add_argument('--model', type=str, default='cnn', help='which model to use')
        parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients")

        parser.add_argument('--N_client', type=int, default=100, help="number of clients")
        parser.add_argument('--local_lr', type=float, default=0.01, help="learning rate")

        args = parser.parse_args()

        self.forget_clients_num = args.forget_clients_num
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch
        self.finetune_epoch = args.finetune_epoch
        self.mix_epoch = args.mix_epoch
        self.rouc = args.rouc
        self.K = args.K
        self.M = args.M
        self.opt = args.opt
        self.p = args.p
        self.n_data = args.n_data
        self.n_data_test = args.n_data_test
        self.n_data_val = args.n_data_val
        self.overlap = args.overlap
        self.model = args.model
        self.frac = args.frac
        self.N_client = args.N_client
        self.local_lr=args.local_lr
        self.data_name=args.data_name


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    FL_params.parse_args()
    torch.manual_seed(FL_params.seed)

    # kwargs for data loader
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))
    # print('We set rouc = ', FL_params.rouc, ', K = ', FL_params.K, ' and M = ', FL_params.M, ' with ',
    #       FL_params.global_epoch, 'epochs fine tuning ')
    print('We are going to forget ', FL_params.forget_clients_num, 'client')
    if FL_params.pretrained:
        print('We use pretrained model: ', FL_params.pretrained_gms_file)

    filename = 'results_ftmix_fixedglobal_lastfinal'
    filexist = os.path.isfile('save/' + filename)
    if (not filexist):
        with open('save/' + filename, 'a') as f1:
            f1.write(
                'dataset;model;epochs;local_ep;num_clients;iid;p;opt;n_data;frac;lr;train_gate_only;val_acc_avg_e2e;val_acc_avg_e2e_neighbour;val_acc_avg_locals;val_acc_avg_fedavg;ft_val_acc;val_acc_avg_3;val_acc_avg_rep;val_acc_avg_repft;ft_train_acc;train_acc_avg_locals;val_acc_avg_gateonly;localtest_acc_fedavg;localtest_acc_local;localtest_acc_ft;localtest_acc_e2e;test_acc_fedavg;test_acc_local;test_acc_ft;test_acc_e2e;overlap;alpha;freeze;run')

            f1.write('\n')
    writer = SummaryWriter(comment=f'lr_{FL_params.local_lr}_p_{FL_params.p}_opt_{FL_params.opt}')

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    # train_frac = FL_params.n_data / (FL_params.n_data + FL_params.n_data_val)

    # 10 indexes
    # selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client+FL_params.K, replace=False)
    # selected_clients = np.random.choice(FL_params.N_total_client, size=int(FL_params.N_client))
    client_loaders = list()
    old_client_models = {'local': [], 'global': [], 'gate': []}
    kwargs = {'num_workers': 1, 'pin_memory': True} if FL_params.cuda_state else {}

    # init_global_model = model_init(FL_params.data_name, device)
    # client_all_loaders, test_loader = dataloader_init(FL_params)
    if FL_params.mix_experts:
        if FL_params.data_name == 'mnist' and FL_params.model == 'cnn':
            dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test = splitExpertData(FL_params)
            # print(type(dict_users))
            # print(dict_users)
            net_glob_fedAvg, gate_model, net_locals, client_models = splittExpertModel(FL_params)
        elif FL_params.data_name == 'cifar10' and FL_params.model == 'cnn':
            dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test = splitExpertData(FL_params)
            # print(type(dict_users))
            # print(dict_users)

            # 创建 DataLoader
            user_train_dataloaders = create_user_dataloaders(dataset_train, dict_users, batch_size=FL_params.local_batch_size)
            user_val_dataloaders = create_user_dataloaders(dataset_train, dict_users_val,
                                                             batch_size=FL_params.local_batch_size)
            user_test_dataloaders = create_user_dataloaders(dataset_test, dict_users_test,
                                                             batch_size=FL_params.local_batch_size)
            client_loaders = user_train_dataloaders
            for user_id, loader in user_train_dataloaders.items():
                print(f"User {user_id} has {len(loader)} batches in train_loader.")
            print('User_test_loader', len(user_test_dataloaders), type(user_test_dataloaders),type(user_test_dataloaders[0]))
            net_glob_fedAvg, gate_model, net_locals, client_models = splittExpertModel(FL_params)

        test_loader = user_test_dataloaders #torch.utils.data.DataLoader(dataset_test, batch_size=FL_params.local_batch_size, shuffle=False,
                                                  # **kwargs)
        # client_loaders = []
        # for ii in range(FL_params.N_total_client):
        #     client_loaders.append(
        #         torch.utils.data.DataLoader(dataset_train, FL_params.local_batch_size, shuffle=True, **kwargs))
    else:
        init_global_model = model_init(FL_params.data_name, FL_params.device)
        client_all_loaders, test_loader = dataloader_init(FL_params)

    # img_size = dataset_train[0][0].shape
    #
    # input_length = 1
    # for x in img_size:
    #     input_length *= x

    # 有自动退出的client，这里默认opt-out=0
    opt_out = [-1] #np.random.choice(range(FL_params.N_client), size=int(FL_params.opt * FL_params.N_client), replace=False)
    # 0.5 *100 =50
    opt_in = [x for x in range(FL_params.N_client) if x not in opt_out]
    print("opt in", opt_in) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # for idx in opt_in:
    #     client_loaders.append(client_all_loaders[idx])
    print("client_loaders len", len(client_loaders)) # 10

    # for idx in selected_clients:
    #     client_loaders.append(client_all_loaders[idx])
    # print('client_loaders', client_loaders)

    # client_all_loaders = client_loaders[selected_clients]
    # client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(FL_params)
    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadowl_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")

    print('len of selected client loader,', len(client_loaders))
    if FL_params.mix_experts:
        # FL
        net_glob_fedAvg, writer = mix_Train(opt_in, dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test,
                                    net_glob_fedAvg,
                                    writer, FL_params)
        print('net_glob_fedAvg', net_glob_fedAvg)

        # TODO finetune client num
        idxs_users = opt_in#np.random.choice(range(FL_params.N_client), 5, replace=False)  # choose users to evaluate on

        print("idxs_users:", idxs_users)

        # Train clients - finetune
        # fintuned: trained global model
        finetuned, locals_nets, val_acc_ft, train_acc_ft, val_acc_locals, train_acc_locals = train_clients(
            idxs_users, dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test, net_glob_fedAvg,
            net_locals, FL_params)
        print('net_locals', net_locals)
        print('locals_nets', locals_nets)
        old_client_models['local'] = copy.deepcopy(locals_nets)
        old_client_models['global'] = copy.deepcopy(finetuned)
        old_client_models['gate'] = copy.deepcopy(gate_model)


        # Mix train clients - mix
        selected_clients, mix_local, mix_global, mix_gate, val_acc_e2e, val_acc_fedavg, mixed_client_models = mix_train_clients(
            idxs_users, dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test, net_glob_fedAvg,
            gate_model, finetuned, locals_nets, FL_params)
        # print('mixed_client_models', mixed_client_models)
        # 保存客户端模型
        new_client_models = copy.deepcopy(mixed_client_models)

        # Test clients
        localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg, test_acc_local, test_acc_ft, test_acc_e2e, test_acc_fedavg = test_clients(
            selected_clients, dataset_train, dict_users, dict_users_val, dataset_test, dict_users_test, net_glob_fedAvg, locals_nets, finetuned, mix_local, mix_global,
            mix_gate, FL_params)

        # Calculate average accuracies
        val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)
        train_acc_avg_locals = sum(train_acc_locals) / len(train_acc_locals)
        val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
        # val_acc_avg_e2e_neighbour = np.nan  # or calculated if applicable
        # val_acc_avg_3 = np.nan  # or calculated if applicable
        # val_acc_avg_gateonly = np.nan  # or calculated if applicable
        # val_acc_avg_rep = np.nan  # or calculated if applicable
        # val_acc_avg_repft = np.nan  # or calculated if applicable
        val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)
        ft_val_acc = sum(val_acc_ft) / len(val_acc_ft)
        ft_train_acc = sum(train_acc_ft) / len(train_acc_ft)

        # Save results
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        learn_file_name = 'learn'+current_time+'.xlsx'
        save_results('learn',FL_params.forget_clients_num, learn_file_name, FL_params, val_acc_avg_e2e, val_acc_avg_locals,
                     val_acc_avg_fedavg, ft_val_acc, ft_train_acc, train_acc_avg_locals,
                     localtest_acc_fedavg, localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, test_acc_fedavg,
                     test_acc_local, test_acc_ft, test_acc_e2e)
        with open('save/'+filename, 'a') as f1:
            f1.write(f'{FL_params.data_name};{FL_params.model};{FL_params.global_epoch};{FL_params.local_epoch};{FL_params.N_client};{FL_params.iid};{FL_params.p};{FL_params.opt};{FL_params.n_data};{FL_params.frac};{FL_params.local_lr};{FL_params.train_gate_only};{val_acc_avg_e2e};{val_acc_avg_locals};{val_acc_avg_fedavg};{ft_val_acc};{ft_train_acc};{train_acc_avg_locals};{localtest_acc_fedavg};{localtest_acc_local};{localtest_acc_ft};{localtest_acc_e2e};{test_acc_fedavg};{test_acc_local};{test_acc_ft};{test_acc_e2e};{FL_params.overlap};{FL_params.freeze}')
            f1.write("\n")

        # # Finding pop clients
        # selected_client, tf_idf_scores = Class_pruner(old_client_models, client_loaders, FL_params)
        #
        # # selected_client, tf_idf_scores = Class_pruner(net_glob_fedAvg, client_loaders[-1], FL_params)
        # # print('selected_client, tf_idf_scores', selected_client, tf_idf_scores)
        # FL_params.forget_client_idx = selected_client
        # pop_list=[]
        # pop_list.extend(FL_params.forget_client_idx)
        # opt_out = pop_list
        # opt_in = [x for x in range(FL_params.N_client) if x not in opt_out]
        # print('opt in here: ',opt_in)
        #
        # print('#' * 10, "Step4: PFU-MoE", '#' * 10)
        #
        # # 更新 opt_in 后的 dict_users
        # new_dict_users = {i: dict_users[i] for i in opt_in}
        # new_dict_users_val = {i: dict_users_val[i] for i in opt_in}
        # new_dict_users_test = {i: dict_users_test[i] for i in opt_in}
        # print(len(new_dict_users),len(new_dict_users_val),len(new_dict_users_test))
        #
        # client_models=old_client_models[0]
        #
        # new_client_models = {
        #     'local': [],
        #     'global': [],
        #     'gate': []
        # }
        #
        # for idx in opt_in:
        #     # Fine-tune the model for the new client
        #     # print('Test for client models to be update after forget',client_models)
        #     fine_tuned_model = client_models['local'][idx]  # Use the existing local model
        #     client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
        #                           idxs_train=new_dict_users[idx], idxs_val=new_dict_users_val[idx],
        #                           idxs_test=new_dict_users_test[idx])
        #     wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
        #         net=copy.deepcopy(fine_tuned_model).to(FL_params.device), n_epochs=FL_params.finetune_epoch, learning_rate=5e-5, val=True)
        #     fine_tuned_model.load_state_dict(wt)
        #     client_models['local'][idx] = fine_tuned_model  # Update the fine-tuned model
        #
        #     # new_client_models.append(client_models['local'][idx])
        #
        #     # Fine-tune the global model for the new client
        #     fine_tuned_global_model = client_models['global'][idx]  # Use the existing global model
        #     wt_global, _, _, _ = client.train_finetune(
        #         net=copy.deepcopy(fine_tuned_global_model).to(FL_params.device), n_epochs=FL_params.finetune_epoch,
        #         learning_rate=5e-5, val=True)
        #     fine_tuned_global_model.load_state_dict(wt_global)
        #     client_models['global'][idx] = fine_tuned_global_model  # Update the fine-tuned global model
        #     # new_client_models['global'].append(fine_tuned_global_model)
        #     # new_client_models.append(client_models['global'][idx])
        #
        #     # Fine-tune the gate model for the new client
        #     fine_tuned_gate_model = client_models['gate'][idx]  # Use the existing gate model
        #     wt_gate, _, _, _ = client.train_finetune(
        #         net=copy.deepcopy(fine_tuned_gate_model).to(FL_params.device), n_epochs=FL_params.finetune_epoch,
        #         learning_rate=5e-5, val=True)
        #     fine_tuned_gate_model.load_state_dict(wt_gate)
        #     client_models['gate'][idx] = fine_tuned_gate_model  # Update the fine-tuned gate model
        #     # new_client_models['gate'].append(fine_tuned_gate_model)
        #     # new_client_models.append(client_models['gate'][idx])
        #     new_client_models['local'].append(client_models['local'][idx])
        #     new_client_models['global'].append(client_models['global'][idx])
        #     new_client_models['gate'].append(client_models['gate'][idx])
        #
        # # Perform unlearning step once with the updated client models
        # print(5 * "#" + "  Calibration Start" + 5 * "#")
        # std_time = time.time()
        # forget_client_idx = FL_params.forget_client_idx
        # old_client_models=old_client_models[0]
        #
        # # 从 old_client_models 中移除指定索引的模型
        # # Ensure forget_client_idx is a list
        # if isinstance(FL_params.forget_client_idx, int):
        #     forget_client_idx = [FL_params.forget_client_idx]
        # elif isinstance(FL_params.forget_client_idx, list):
        #     forget_client_idx = FL_params.forget_client_idx
        # else:
        #     raise ValueError("forget_client_idx is not valid")
        #
        # # # Removing the clients from old_client_models
        # # for key in old_client_models:
        # #     if isinstance(old_client_models[key], list):
        # #         for idx in sorted(forget_client_idx, reverse=True):  # Reverse to avoid index shifting
        # #             if idx < len(old_client_models[key]):
        # #                 old_client_models[key].pop(idx)
        # #             else:
        # #                 raise IndexError(f"Index {idx} out of range for {key}")
        #
        # # Removing the clients from old_client_models['local'], ['global'], and ['gate']
        # for idx in sorted(forget_client_idx, reverse=True):  # Reverse to avoid index shifting
        #     if idx < len(old_client_models['local']):
        #         old_client_models['local'].pop(idx)
        #     if idx < len(old_client_models['global']):
        #         old_client_models['global'].pop(idx)
        #     if idx < len(old_client_models['gate']):
        #         old_client_models['gate'].pop(idx)
        #
        #
        # for idx in range(len(new_client_models['local'])):
        #     old_local = old_client_models['local'][idx]
        #     old_global = old_client_models['global'][idx]
        #     old_gate = old_client_models['gate'][idx]
        #
        #     new_local = new_client_models['local'][idx]
        #     new_global = new_client_models['global'][idx]
        #     new_gate = new_client_models['gate'][idx]
        # #     # 校准本地模型
        # #     new_client_models['local'][idx] = unlearning_step_once(old_local, new_local, net_glob_fedAvg,
        # #                                                            new_client_models['local'])
        #
        #     # # 校准全局模型
        #     # new_client_models['global'][idx] = unlearning_step_once(old_global, new_global, net_glob_fedAvg,
        #     #                                                         fine_tuned_model[idx]['global'])
        #     #
        #     # # 校准门控模型
        #     # new_client_models['gate'][idx] = unlearning_step_once(old_gate, new_gate, net_glob_fedAvg,
        #     #                                                       fine_tuned_model[idx]['gate'])
        # # for c in range(len(new_client_models)):
        # # print(old_client_models)
        # # print(new_client_models)
        # # print(len(old_client_models),len(new_client_models))
        # new_global_model = unlearning_step_once(old_local, new_local, old_global, new_global)
        # end_time = time.time()
        # time_learn = (std_time - end_time)
        # print(" Calibration time consuming = {} secods".format(-time_learn))
        #
        # print(5 * "#" + "  Calibration End" + 5 * "#")
        # # 更新后的 clients
        # finetuned, locals_nets, mix_local, mix_global, mix_gate = update_clients(opt_in, dataset_train, dataset_test,
        #                                                                          new_dict_users, new_dict_users_val,
        #                                                                          new_dict_users_test, net_glob_fedAvg,
        #                                                                          net_locals, gate_model, FL_params)
        #
        # # Save results after unlearning step
        # localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg1, test_acc_local, test_acc_ft, test_acc_e2e, test_acc_fedavg = test_clients(
        #     opt_in, dataset_train, new_dict_users, new_dict_users_val, dataset_test, new_dict_users_test, new_global_model, locals_nets, finetuned, mix_local, mix_global,
        #     mix_gate, FL_params)
        # # finetuned, locals_nets, mix_local, mix_global, mix_gate = update_clients(
        # #     idxs_users, dataset_train, dataset_test, new_dict_users, new_dict_users_val, new_dict_users_test,
        # #     net_glob_fedAvg, net_locals, gate_model, FL_params)
        # #
        # # # Test clients
        # # localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg, test_acc_local, test_acc_ft, test_acc_e2e, test_acc_fedavg = test_clients(
        # #     idxs_users, dataset_train, dict_users, dict_users_val, dataset_test, dict_users_test,
        # #     net_glob_fedAvg, locals_nets, finetuned, mix_local, mix_global, mix_gate, FL_params)
        #
        # # Calculate average accuracies after unlearning step
        # # Ensure variables are defined
        # # val_acc_locals = []
        # # train_acc_locals = []
        # # val_acc_e2e = []
        # # val_acc_fedavg = []
        # # val_acc_ft = []
        # # train_acc_ft = []
        # # localtest_acc_fedavg = []
        # # localtest_acc_local = []
        # # localtest_acc_ft = []
        # # localtest_acc_e2e = []
        # # test_acc_fedavg = []
        # # test_acc_local = []
        # # test_acc_ft = []
        # # test_acc_e2e = []
        # # Assuming these variables are calculated in the process
        # val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)
        # train_acc_avg_locals = sum(train_acc_locals) / len(train_acc_locals)
        # val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
        # val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)
        # ft_val_acc = sum(val_acc_ft) / len(val_acc_ft)
        # ft_train_acc = sum(train_acc_ft) / len(train_acc_ft)
        #
        # current_time1 = datetime.now().strftime('%Y%m%d_%H%M%S')
        # unlearn_file_name = 'unlearn' + current_time1 + '.xlsx'
        #
        # # save_results('unlearn', unlearn_file_name, FL_params, val_acc_avg_e2e, val_acc_avg_locals,
        # #              val_acc_avg_fedavg, ft_val_acc, ft_train_acc, train_acc_avg_locals,
        # #              localtest_acc_fedavg, localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, test_acc_fedavg,
        # #              test_acc_local, test_acc_ft, test_acc_e2e)
        # save_results('unlearn',FL_params.forget_clients_num, unlearn_file_name, FL_params, val_acc_avg_e2e, val_acc_avg_locals,
        #              val_acc_avg_fedavg, ft_val_acc, ft_train_acc, train_acc_avg_locals,
        #              localtest_acc_fedavg1, localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, test_acc_fedavg,
        #              test_acc_local, test_acc_ft, test_acc_e2e)


    else:
        old_GMs, unlearn_GMs, old_CMs = federated_learning_unlearning(init_global_model,
                                                                      client_loaders,
                                                                      test_loader,
                                                                      FL_params)
        # """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60 * '=')
    print("Step5. Membership Inference Attack aganist GM...")
    # Perform membership inference attack
    print(client_loaders)
    # print(test_loader)  # len==10
    print(old_client_models)
    # client_loaders1 = {idx: [dataset_train[i] for i in dict_users[idx]] for idx in selected_clients}
    # # test_loader1 = [dataset_test[i] for i in range(len(dataset_test))]  # Assuming test_loader is a list of test samples
    # test_loader1 = user_test_dataloaders  # Assuming test_loader is a dictionary of test samples

    ACC_old, PRE_old, ACC_retrain, PRE_retrain, ACC_unlearn, PRE_unlearn = membership_inference_attack(
        old_client_models, new_client_models, client_loaders, test_loader, FL_params)

    # Print results
    print(f"ACC_old: {ACC_old}, PRE_old: {PRE_old}")
    if FL_params.if_retrain:
        print(f"ACC_retrain: {ACC_retrain}, PRE_retrain: {PRE_retrain}")
    print(f"ACC_unlearn: {ACC_unlearn}, PRE_unlearn: {PRE_unlearn}")
    # if FL_params.if_retrain == True:
    #
    #     t1 = time.time()
    #
    #     unlearn_GMs = unlearn_GMs[-1].to(device)
    #     retrain_GMs = FL_Retrain(unlearn_GMs, client_loaders, test_loader, FL_params)
    #
    #     t2 = time.time()
    #     print("Retrain Time using = {} seconds".format(t2 - t1), 3)

    # '''
    # Saving ckpt
    # '''
    # # prune_model_save_dir = "ckpt/prune"
    # # prune_model_path=os.path.join(prune_model_save_dir, f"prune_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    # # torch.save(unlearn_GMs.state_dict(), prune_model_path)
    #
    # retrain_model_save_dir = "ckpt/retrain"
    # retrain_model_path = os.path.join(retrain_model_save_dir,
    #                                   f"retrain_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    # # torch.save(retrain_GMs[-1].state_dict(), retrain_model_path)
    #
    # finetuned_model_save_dir = "ckpt/finetuned"
    # finetuned_model_path = os.path.join(finetuned_model_save_dir,
    #                                     f"retrain_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    # torch.save(unlearn_GMs.state_dict(), finetuned_model_path)

    '''
     Evaluation
     '''
    # fedavg_test_acc, fedavg_test_loss = test(old_GMs, test_loader, FL_params)
    # # fedaccum_test_acc, fedaccum_test_loss = test(uncali_unlearn_GMs[-1], test_loader)#[-1], test_loader)
    # # fedretrain_test_acc, fedretrain_test_loss = test(retrain_GMs[-1], test_loader,FL_params)
    # fedmoe_test_acc, fedmoe_test_loss = test(unlearn_GMs, test_loader, FL_params)

    # # FL_params.unlearn_client=int(FL_params.forget_client_idx)
    # print('FL_params.unlearn_client: ', FL_params.forget_client_idx)
    # for k in FL_params.forget_client_idx:
    #     target_loader = client_loaders[k]
    #     fedmoe_target_acc, fedmoe_target_loss = test(unlearn_GMs, target_loader, FL_params)
    # # fedaccum_target_acc, fedaccum_target_loss = test(uncali_unlearn_GMs[-1], target_loader)
    # # fedretrain_target_acc, fedretrain_target_loss = test(retrain_GMs[-1], target_loader,FL_params)
    # fedavg_target_acc, fedavg_target_loss = test(old_GMs, target_loader, FL_params)
    #
    # print(5 * "*" + "  Result Summary  " + 5 * "*")
    # print("[FedPruner] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_test_loss,
    #                                                                                  fedmoe_test_acc))
    # # print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss,
    # #                                                                                 fedaccum_test_acc))
    # # print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss,
    # #                                                                                   fedretrain_test_acc))
    # print(
    #     "[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    # print("\n")
    # print("[FedPruner] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_target_loss,
    #                                                                                    fedmoe_target_acc))
    # # print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss,
    # #                                                                                   fedaccum_target_acc))
    # # print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss,
    # #                                                                                     fedretrain_target_acc))
    # print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss,
    #                                                                                 fedavg_target_acc))
    #TODO MIA
    # """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""
    #

    #
    # T_epoch = -1
    # # MIA setting:Target model == Shadow Model
    # T_epoch=0
    # old_GM = old_client_models[T_epoch] #old_GMs  # [T_epoch]
    # print(len(test_loader), type(test_loader))
    # attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)
    #
    # print("\nEpoch  = {}".format(T_epoch))
    # print("Attacking against FL Standard  ")
    # target_model = old_GMs  # [T_epoch]
    # (ACC_old, PRE_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    #
    # if (FL_params.if_retrain == True):
    #     print("Attacking against FL Retrain  ")
    #     # target_model = retrain_GMs[T_epoch]
    #     (ACC_retrain, PRE_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    #
    # print("Attacking against FL Client Unlearn  ")
    # target_model = unlearn_GMs  # [T_epoch]
    # (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('fork')
    Federated_Unlearning()
