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

# ourself libs
# from model_initiation import model_init
from data_preprocess import data_set, dataloader_init, model_init, Net_cifar10, Net_mnist
from FL_base import test, FL_Retrain

from Fed_Unlearn_base import federated_learning_unlearning
from membership_inference import train_attack_model, attack

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

"""Step 0. Initialize Federated Unlearning parameters"""

class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10  ## Total number of clients N.
        # self.data_name = 'cifar10'
        self.data_name = 'shakespeare'# purchase, cifar10, mnist, adult

        self.pretrained = True#False
        # self.pretrained_gms_file = 'cifar_global.pth'#'shake20_global.pth'
        # self.pretrained_cms_file = 'cifar_client.pth'#'shake20_client.pth'
        self.pretrained_gms_file = 'shake_20_global.pth'
        self.pretrained_cms_file = 'shake_20_client.pth'
        self.save_pretrained = False#False

        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = -1  # If want to forget, change None to the client index
        self.forget_clients_num=1 #1-10

        self.global_epoch = 20  # 600#20#600  # T
        self.local_epoch = 2  # 10#2#10 # E

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
        self.K = 3
        self.b = 1
        self.N_datapoint = 20  # 80 #16
        self.M = 3000#100  # 600
        self.if_sample_unlearning = False
        self.selected_K_group = []
        self.sparsity = 0.05

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Federated Unlearning Arguments')
        parser.add_argument('--forget_clients_num', type=int, default=self.forget_clients_num,
                            help='Number of clients to forget')
        parser.add_argument('--global_epoch', type=int, default=self.global_epoch, help='Number of global epochs')
        parser.add_argument('--local_epoch', type=int, default=self.local_epoch, help='Number of local epochs')
        parser.add_argument('--rouc', type=float, default=self.rouc, help='RouC value')
        parser.add_argument('--K', type=int, default=self.K, help='K value')
        parser.add_argument('--M', type=int, default=self.M, help='M value')
        args = parser.parse_args()

        self.forget_clients_num = args.forget_clients_num
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch
        self.rouc = args.rouc
        self.K = args.K
        self.M = args.M


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
    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_global_model = model_init(FL_params.data_name, device)
    client_all_loaders, test_loader = dataloader_init(FL_params)

    # 10 indexes
    # selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client+FL_params.K, replace=False)
    selected_clients = np.random.choice(FL_params.N_total_client, size=int(FL_params.N_client))
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])

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

    old_GMs, unlearn_GMs, old_CMs = federated_learning_unlearning(init_global_model,
                                                                  client_loaders,
                                                                  test_loader,
                                                                  FL_params)
    if FL_params.if_retrain == True:

        t1 = time.time()

        unlearn_GMs = unlearn_GMs[-1]
        retrain_GMs = FL_Retrain(unlearn_GMs, client_loaders, test_loader, FL_params)

        t2 = time.time()
        print("Retrain Time using = {} seconds".format(t2 - t1), 3)

    '''
     Evaluation
     '''
    fedavg_test_acc, fedavg_test_loss = test(old_GMs[-1], test_loader,FL_params)
    # fedaccum_test_acc, fedaccum_test_loss = test(uncali_unlearn_GMs[-1], test_loader)#[-1], test_loader)
    fedretrain_test_acc, fedretrain_test_loss = test(retrain_GMs[-1], test_loader,FL_params)
    fedmoe_test_acc, fedmoe_test_loss = test(unlearn_GMs, test_loader,FL_params)

    # FL_params.unlearn_client=int(FL_params.forget_client_idx)
    print('FL_params.unlearn_client: ', FL_params.forget_client_idx)
    for k in FL_params.forget_client_idx:
        target_loader = client_loaders[k]
        fedmoe_target_acc, fedmoe_target_loss = test(unlearn_GMs, target_loader,FL_params)
    # fedaccum_target_acc, fedaccum_target_loss = test(uncali_unlearn_GMs[-1], target_loader)
    fedretrain_target_acc, fedretrain_target_loss = test(retrain_GMs[-1], target_loader,FL_params)
    fedavg_target_acc, fedavg_target_loss = test(old_GMs[-1], target_loader,FL_params)




    print(5 * "*" + "  Result Summary  " + 5 * "*")
    print("[FedPruner] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_test_loss,
                                                                                     fedmoe_test_acc))
    # print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss,
    #                                                                                 fedaccum_test_acc))
    print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss,
                                                                                      fedretrain_test_acc))
    print(
        "[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    print("\n")
    print("[FedPruner] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedmoe_target_loss,
                                                                                       fedmoe_target_acc))
    # print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss,
    #                                                                                   fedaccum_target_acc))
    print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss,
                                                                                        fedretrain_target_acc))
    print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss,
                                                                                    fedavg_target_acc))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60 * '=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = old_GMs[T_epoch]
    (ACC_old, PRE_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    if (FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        target_model = retrain_GMs[T_epoch]
        (ACC_retrain, PRE_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Client Unlearn  ")
    target_model = unlearn_GMs  # [T_epoch]
    (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)



if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('fork')
    Federated_Unlearning()

