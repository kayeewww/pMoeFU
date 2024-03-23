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

# ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test

from Fed_Unlearn_base import federated_learning_unlearning

# from membership_inference import train_attack_model, attack

"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 25
        self.data_name = 'cifar10'  # cifar10, cifar100
        self.model_name = 'resnet44'  # resnet20, resnet32, resnet44, resnet56, vgg11, vgg13, vgg16, vgg19
        self.global_epoch = 2000
        self.local_epoch = 5

        self.save_acc = 80
        self.save_re_acc = 75

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = True
        self.model_file = 'seed_acc80.06_epoch125_2024-03-02 15-08-23.pth'

        # 要剪枝的类别。
        self.unlearn_class = 2  # If want to forget, change None to the client index
        self.sparsity = 0.05
        # If this parameter is set to False, only the global model after the final training is completed is output
        # 如果设置为 True，表示在遗忘操作后执行重新训练。在重新训练期间，与遗忘的客户端相关的数据将被丢弃
        self.if_retrain = False  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.
        # 如果设置为 True，表示在训练期间跳过需要被遗忘的用户。如果设置为 False，则 global_train_once 函数不会跳过需要被遗忘的用户
        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5  # When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
        # forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)
    # kwargs for data loader
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    init_global_model = model_init(FL_params.data_name, FL_params.model_name)
    client_all_loaders, test_loader = data_init(FL_params)
    print(init_global_model)

    # 从100个客户端中随机抽取25个作为实验对象
    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])

    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadow_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params)


if __name__ == '__main__':
    Federated_Unlearning()
