import torch
import numpy as np
import time
import argparse
import os
import copy
from torchvision import datasets, transforms

# ourselves libs
# from model_initiation import model_init
from data_preprocess import data_set, dataloader_init, model_init, Net_cifar10, Net_mnist, splitExpertData, splittExpertModel
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

def mix_fl_finetune(idxs_users,dataset_train,dataset_test,dict_users,dict_users_val,dict_users_test,net_glob_fedAvg,gate_model,net_locals,filename,FL_params):
    val_acc_locals, val_acc_mix, val_acc_fedavg, val_acc_e2e, val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft, val_acc_e2e_neighbour, val_acc_gateonly = [], [], [], [], [], [], [], [], [], []
    train_acc_ft, train_acc_locals = [], []
    acc_test_l, acc_test_m = [], []
    gate_values = []
    finetuned = []
    locals_nets = []
    # idxs_users = np.random.choice(range(FL_params.N_client), 5, replace=False)  # choose users to evaluate on
    for idx in idxs_users:
        client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
                              idxs_train=dict_users[idx],
                              idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx])

        # finetune FedAvg for every client
        print("Finetune %d" % (idx))
        wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
            net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), n_epochs=500, learning_rate=5e-5, val=True)
        val_acc_ft.append(val_acc_finetuned)
        train_acc_ft.append(train_acc_finetuned)

        ft_net = copy.deepcopy(net_glob_fedAvg)
        ft_net.load_state_dict(wt)
        finetuned.append(ft_net)

        # train local model
        print("Local %d" % (idx))
        net_local_idx = copy.deepcopy(net_locals).to(FL_params.device)
        w_l, _, val_acc_l, train_acc_l = client.train_finetune(net=net_local_idx, n_epochs=500, learning_rate=5e-5,
                                                               val=True)

        net_local_idx.load_state_dict(w_l)
        locals_nets.append(net_local_idx)

        val_acc_locals.append(val_acc_l)
        train_acc_locals.append(train_acc_l)
    mix_local = []
    mix_global = []
    mix_gate = []
    for i, idx in enumerate(idxs_users):
        client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
                              idxs_train=dict_users[idx],
                              idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx])

        print("E2e %d" % (idx))
        gate_idx = copy.deepcopy(gate_model).to(FL_params.device)
        # initialize gate with global model conv parameters
        # if (args.freeze):
        if (1):
            gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
            gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
            ct = 0
            for child in gate_idx.children():
                ct += 1
                if (ct < 4):
                    for param in child.parameters():
                        param.requiers_grad = False
            ct = 0
            # freeze conv layers for finetuned model
            for child in finetuned[i].children():
                ct += 1
                if (ct < 4):
                    for param in child.parameters():
                        param.requires_grad = False

        gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(
            net_local=copy.deepcopy(finetuned[i]).to(FL_params.device),
            net_global=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), gate=gate_idx,
            train_gate_only=FL_params.train_gate_only, n_epochs=500, early_stop=True, learning_rate=FL_params.local_lr,
            val=True)

        gate_idx.load_state_dict(gate_w)
        mix_gate.append(copy.deepcopy(gate_idx))

        mix_l = copy.deepcopy(net_locals)
        mix_g = copy.deepcopy(net_glob_fedAvg)

        mix_l.load_state_dict(local_w)
        mix_g.load_state_dict(global_w)

        mix_local.append(mix_l)
        mix_global.append(mix_g)

        val_acc_e2e.append(val_acc_e2e_k)

        # evaluate FedAvg on local dataset
        val_acc_fed, _ = client.validate(net=net_glob_fedAvg.to(FL_params.device), val=True)
        val_acc_fedavg.append(val_acc_fed)
        # evaluate with local client test set
    localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg = [], [], [], []
    for i, idx in enumerate(idxs_users):
        client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
                              idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
                              idxs_test=dict_users_test[idx])

        localtest_acc_e2e_idx, _ = client.validate_mix(net_l=mix_local[i],
                                                       net_g=mix_global[i],
                                                       gate=mix_gate[i], val=False)

        localtest_acc_e2e.append(localtest_acc_e2e_idx)

        localtest_acc_fedavg_idx, _ = client.validate(net=net_glob_fedAvg, val=False)
        localtest_acc_fedavg.append(localtest_acc_fedavg_idx)

        localtest_acc_local_idx, _ = client.validate(net=locals_nets[i], val=False)
        localtest_acc_local.append(localtest_acc_local_idx)

        localtest_acc_ft_idx, _ = client.validate(net=finetuned[i], val=False)
        localtest_acc_ft.append(localtest_acc_ft_idx)
    localtest_acc_local = sum(localtest_acc_local) / len(localtest_acc_local)
    localtest_acc_ft = sum(localtest_acc_ft) / len(localtest_acc_ft)
    localtest_acc_e2e = sum(localtest_acc_e2e) / len(localtest_acc_e2e)
    localtest_acc_fedavg = sum(localtest_acc_fedavg) / len(localtest_acc_fedavg)

    # evaluate all models on balanced (global) dataset
    test_acc_local, test_acc_ft, test_acc_e2e = [], [], []
    print("testing FedAvg...")
    test_acc_fedavg, _ = test_img(net_glob_fedAvg, dataset_test, FL_params)
    for i, idx in enumerate(idxs_users):
        print(idx)
        print("testing Locals...")
        test_acc_local_idx, _ = test_img(locals_nets[i], dataset_test, FL_params)
        print("testing Finetune...")
        test_acc_ft_idx, _ = test_img(finetuned[i], dataset_test, FL_params)
        print("testing mixture...")
        test_acc_e2e_idx, _ = test_img_mix(mix_local[i], mix_global[i], mix_gate[i], dataset_test, FL_params)

        test_acc_local.append(test_acc_local_idx)
        test_acc_ft.append(test_acc_ft_idx)
        test_acc_e2e.append(test_acc_e2e_idx)

    test_acc_local = sum(test_acc_local) / len(test_acc_local)
    test_acc_ft = sum(test_acc_ft) / len(test_acc_ft)
    test_acc_e2e = sum(test_acc_e2e) / len(test_acc_e2e)

    # Calculate validation and test accuracies

    val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)
    # val_acc_avg_locals = np.nan

    train_acc_avg_locals = sum(train_acc_locals) / len(train_acc_locals)
    # train_acc_avg_locals = np.nan

    val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
    # val_acc_avg_e2e = np.nan

    # val_acc_avg_e2e_neighbour = sum(val_acc_e2e_neighbour) / len(val_acc_e2e_neighbour)
    val_acc_avg_e2e_neighbour = np.nan

    # val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
    val_acc_avg_3 = np.nan

    # val_acc_avg_gateonly = sum(val_acc_gateonly) / len(val_acc_gateonly)
    val_acc_avg_gateonly = np.nan

    # val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
    val_acc_avg_rep = np.nan

    # val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
    val_acc_avg_repft = np.nan

    val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)

    ft_val_acc = sum(val_acc_ft) / len(val_acc_ft)
    # ft_val_acc = np.nan

    ft_train_acc = sum(train_acc_ft) / len(train_acc_ft)
    # ft_train_acc = np.nan


    with open('save/' + filename, 'a') as f1:
        f1.write(
            f'{FL_params.data_name};{FL_params.model};{FL_params.global_epoch};{FL_params.local_epoch};{FL_params.N_client};{FL_params.iid};{FL_params.p};{FL_params.opt};{FL_params.n_data};{FL_params.frac};{FL_params.local_lr};{FL_params.train_gate_only};{val_acc_avg_e2e};{val_acc_avg_e2e_neighbour};{val_acc_avg_locals};{val_acc_avg_fedavg};{ft_val_acc};{val_acc_avg_3};{val_acc_avg_rep};{val_acc_avg_repft};{ft_train_acc};{train_acc_avg_locals};{val_acc_avg_gateonly};{localtest_acc_fedavg};{localtest_acc_local};{localtest_acc_ft};{localtest_acc_e2e};{test_acc_fedavg};{test_acc_local};{test_acc_ft};{test_acc_e2e};{FL_params.overlap}')
        f1.write("\n")




    return net_glob_fedAvg, locals_nets