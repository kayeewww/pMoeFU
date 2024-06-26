# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
# %%
import torch
import numpy as np
import time
import argparse
import os
import copy
from torchvision import datasets, transforms

# ourself libs
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

from mix_fl import mix_fl_finetune


warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

"""Step 0. Initialize Federated Unlearning parameters"""

class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10  ## Total number of clients N.
        self.data_name = 'cifar10'
        self.model='cnn'
        self.mix_experts_modelset = ['mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist', 'mix_cifar10', 'mix_mnist']
        self.mix_experts = True
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.data_name = 'shakespeare'# purchase, cifar10, mnist, adult

        self.pretrained = False
        self.pretrained_gms_file = 'mix_global.pth'#'ssh20_cifar_global.pth'#'shake20_global.pth'
        self.pretrained_cms_file = 'mix_client.pth'#'ssh20_cifar_client.pth'#'shake20_client.pth'
        # self.pretrained_gms_file = 'shake20_global.pth'
        # self.pretrained_cms_file = 'shake20_client.pth'
        self.save_pretrained = False#False

        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 1  # If want to forget, change None to the client index
        self.forget_clients_num = 1 #1-10

        self.global_epoch = 20  # 600#20#600  # T
        self.local_epoch = 2  # 10#2#10 # E
        self.finetune_epoch = 2

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
        self.if_retrain = True  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5
        self.client_fraction = 10  # Fraction of clients selected per round.

        self.fats_method = 'client'
        self.rouc = 0.3#0.2
        self.rous = 0.1
        self.k_u = -1
        self.unlearn_client = -1
        self.sparsity = 0.05
        self.class_flag = False
        self.rest_data_loader = None  # torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        self.rest_testdata = None
        self.K = 2
        self.b = 1
        self.N_datapoint = 20  # 80 #16
        self.M = 2000#100  # 600
        self.if_sample_unlearning = False
        self.selected_K_group = []
        self.sparsity = 0.05
        self.tv_stability_threshold=2#0.01


        self.p=0.3
        self.opt=0.5
        self.n_data=100
        self.n_data_test=200
        self.n_data_val=200
        self.overlap=True
        self.frac=0.1
        self.train_gate_only=True
        self.iid=False


    def parse_args(self):
        parser = argparse.ArgumentParser(description='Federated Unlearning Arguments')
        parser.add_argument('--forget_clients_num', type=int, default=self.forget_clients_num,
                            help='Number of clients to forget')
        parser.add_argument('--global_epoch', type=int, default=self.global_epoch, help='Number of global epochs')
        parser.add_argument('--local_epoch', type=int, default=self.local_epoch, help='Number of local epochs')
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

        args = parser.parse_args()

        self.forget_clients_num = args.forget_clients_num
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch
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


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    FL_params.parse_args()
    torch.manual_seed(FL_params.seed)

    # kwargs for data loader
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))
    print('We set rouc = ',FL_params.rouc,', K = ', FL_params.K,' and M = ',FL_params.M,' with ',FL_params.global_epoch,'epochs fine tuning ')
    print('We are going to forget ',FL_params.forget_clients_num, 'client')
    if FL_params.pretrained:
        print('We use pretrained model: ',FL_params.pretrained_gms_file)

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

    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # train_frac = FL_params.n_data / (FL_params.n_data + FL_params.n_data_val)
    # 10 indexes
    # selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client+FL_params.K, replace=False)
    selected_clients = np.random.choice(FL_params.N_total_client, size=int(FL_params.N_client))
    client_loaders = list()
    kwargs = {'num_workers': 1, 'pin_memory': True} if FL_params.cuda_state else {}

    # init_global_model = model_init(FL_params.data_name, device)
    # client_all_loaders, test_loader = dataloader_init(FL_params)
    if FL_params.mix_experts:
        if FL_params.data_name == 'mnist' and FL_params.model == 'cnn':
            dataset_train, dataset_test, dict_users=splitExpertData(FL_params)
            print(type(dict_users))
            print(dict_users)
            net_glob_fedAvg, gate_model, net_locals=splittExpertModel(FL_params)
        elif FL_params.data_name == 'cifar10' and FL_params.model == 'cnn':
            dataset_train,dataset_test, dict_users, dict_users_val, dict_users_test=splitExpertData(FL_params)
            print(type(dict_users))
            print(dict_users)
            net_glob_fedAvg, gate_model, net_locals=splittExpertModel(FL_params)
        init_global_model = net_glob_fedAvg
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=FL_params.local_batch_size, shuffle=False, **kwargs)
        client_loaders = []
        for ii in range(FL_params.N_total_client):
            client_loaders.append(torch.utils.data.DataLoader(dataset_train, FL_params.local_batch_size, shuffle=True, **kwargs))
        client_all_loaders=client_loaders
        # init_global_model_list = model_init(FL_params.mix_experts_modelset, device)
        # client_all_loaders, test_loader = dataloader_init(FL_params)
    else:
        init_global_model = model_init(FL_params.data_name, FL_params.device)
        client_all_loaders, test_loader = dataloader_init(FL_params)

    img_size = dataset_train[0][0].shape

    input_length = 1
    for x in img_size:
        input_length *= x

    # 有自动退出的client，这里默认opt-out=0
    opt_out = np.random.choice(range(FL_params.N_client), size=int(FL_params.opt * FL_params.N_client), replace=False)
    # 0.5 *100 =50
    opt_in = [x for x in range(FL_params.N_client) if x not in opt_out]
    print("opt in",opt_in)


    for idx in opt_in:
        client_loaders.append(client_all_loaders[idx])

    #TODO
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

    # trainset, testset = data_set(FL_params.data_name)

    # client = CifarClient(trainset, testset, device, init_global_model).to_client()
    # fl.client.start_client(server_address="127.0.0.1:8080", client=client)

    print('len of selected client loader,', len(client_loaders))
    if FL_params.mix_experts:
        net_glob_fedAvg=mix_Train(opt_in, dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test, net_glob_fedAvg,
                  writer, FL_params)
        old_GMs=net_glob_fedAvg
        idxs_users = np.random.choice(range(FL_params.N_client), 10, replace=False)  # choose users to evaluate on

        print("idxs_users:", idxs_users)
        net_glob_fedAvg, locals_nets=mix_fl_finetune(idxs_users,dataset_train,dataset_test,dict_users,dict_users_val,dict_users_test,net_glob_fedAvg,gate_model,net_locals,filename,FL_params)

        selected_client, tf_idf_scores = Class_pruner(net_glob_fedAvg, FL_params)
        print('selected_client, tf_idf_scores', selected_client, tf_idf_scores)
        FL_params.forget_client_idx = selected_client

        print('#' * 10, "Test again for mixture", '#' * 10)
        updated_global_models = list()
        client_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # if 1:#k_u in selected_clients:
        unlearn_CMS = []
        temp = locals_nets
        print(temp)
        if type(FL_params.forget_client_idx) == int:
            temp.pop(FL_params.forget_client_idx)
        else:
            for k_client in FL_params.forget_client_idx:
                temp.pop(k_client)

        # remain_client_list = client_list.pop(k_client)
        # remain_client_list=[]
        client_set = set(client_list)
        forget_set = set(FL_params.forget_client_idx)
        remain_client_list = list(client_set - forget_set)
        print('After Removing', remain_client_list)
        unlearn_CMS.append(temp)
        idxs_users_pop = remain_client_list#[i for i in range(FL_params.N_client) if i != forget_set]
        print('idxs_users_pop',idxs_users_pop)
        unlearn_GMs, locals_nets = mix_fl_finetune(idxs_users_pop, dataset_train, dataset_test, dict_users,
                                                       dict_users_val, dict_users_test, net_glob_fedAvg, gate_model,
                                                       net_locals, filename, FL_params)


    else:
        old_GMs, unlearn_GMs, old_CMs = federated_learning_unlearning(init_global_model,
                                                                  client_loaders,
                                                                  test_loader,
                                                                  FL_params)
    # if FL_params.if_retrain == True:
    #
    #     t1 = time.time()
    #
    #     unlearn_GMs = unlearn_GMs[-1].to(device)
    #     retrain_GMs = FL_Retrain(unlearn_GMs, client_loaders, test_loader, FL_params)
    #
    #     t2 = time.time()
    #     print("Retrain Time using = {} seconds".format(t2 - t1), 3)

    '''
    Saving ckpt
    '''
    # prune_model_save_dir = "ckpt/prune"
    # prune_model_path=os.path.join(prune_model_save_dir, f"prune_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    # torch.save(unlearn_GMs.state_dict(), prune_model_path)

    retrain_model_save_dir="ckpt/retrain"
    retrain_model_path = os.path.join(retrain_model_save_dir,
                                    f"retrain_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    # torch.save(retrain_GMs[-1].state_dict(), retrain_model_path)

    finetuned_model_save_dir = "ckpt/finetuned"
    finetuned_model_path = os.path.join(finetuned_model_save_dir,
                                      f"retrain_{FL_params.rouc}_{FL_params.K}_{FL_params.M}_forget${FL_params.forget_clients_num}clients.pth")
    torch.save(unlearn_GMs.state_dict(), finetuned_model_path)


    '''
     Evaluation
     '''
    fedavg_test_acc, fedavg_test_loss = test(old_GMs, test_loader,FL_params)
    # fedaccum_test_acc, fedaccum_test_loss = test(uncali_unlearn_GMs[-1], test_loader)#[-1], test_loader)
    # fedretrain_test_acc, fedretrain_test_loss = test(retrain_GMs[-1], test_loader,FL_params)
    fedmoe_test_acc, fedmoe_test_loss = test(unlearn_GMs, test_loader,FL_params)

    # FL_params.unlearn_client=int(FL_params.forget_client_idx)
    print('FL_params.unlearn_client: ', FL_params.forget_client_idx)
    for k in FL_params.forget_client_idx:
        target_loader = client_loaders[k]
        fedmoe_target_acc, fedmoe_target_loss = test(unlearn_GMs, target_loader,FL_params)
    # fedaccum_target_acc, fedaccum_target_loss = test(uncali_unlearn_GMs[-1], target_loader)
    # fedretrain_target_acc, fedretrain_target_loss = test(retrain_GMs[-1], target_loader,FL_params)
    fedavg_target_acc, fedavg_target_loss = test(old_GMs, target_loader,FL_params)




    print(5 * "*" + "  Result Summary  " + 5 * "*")
    print("[FedPruner] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_test_loss,
                                                                                     fedmoe_test_acc))
    # print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss,
    #                                                                                 fedaccum_test_acc))
    # print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss,
    #                                                                                   fedretrain_test_acc))
    print(
        "[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    print("\n")
    print("[FedPruner] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_target_loss,
                                                                                       fedmoe_target_acc))
    # print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss,
    #                                                                                   fedaccum_target_acc))
    # print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss,
    #                                                                                     fedretrain_target_acc))
    print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss,
                                                                                    fedavg_target_acc))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60 * '=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs#[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = old_GMs#[T_epoch]
    (ACC_old, PRE_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    if (FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        # target_model = retrain_GMs[T_epoch]
        (ACC_retrain, PRE_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Client Unlearn  ")
    target_model = unlearn_GMs  # [T_epoch]
    (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)



if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('fork')
    Federated_Unlearning()

