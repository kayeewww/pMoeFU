# config.py
import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl
import torch

import utils

import warnings

from flwr_datasets import FederatedDataset
from FL_base import test
from model_initiation import Net_cifar10, model_init
from data_preprocess import data_init, data_init_with_shadow, data_set

class FL_Params:


    # seed = None

    def __init__(self):
        self.K = 2
        self.device = 'cpu'
        #Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10 ## Total number of clients N.
        self.data_name = 'cifar10'# purchase, cifar10, mnist, adult
        self.model_name = Net_cifar10

        self.global_epoch = 20#600  # T
        self.local_epoch = 2#10 # E


        #Model Training Settings
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

        self.model_file = 'seed_acc80.06_epoch125_2024-03-02 15-08-23.pth'


        #Federated Unlearning Settings
        self.unlearn_interval= 1#Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2 #If want to forget, change None to the client index

                                #If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = True#If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False#If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5 #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
                                            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False
        self.client_fraction = 10  # Fraction of clients selected per round.
        # self.fats_method = 'sample'  # client, sample
        self.fats_method = 'client'
        self.rouc = 0.2
        self.rous = 0.1
        self.k_u=-1
        self.unlearn_client=-1
        self.sparsity = 0.05
        self.class_flag = False
        self.rest_data_loader = None  # torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        self.rest_testdata = None
        self.K = 2
        self.b = 1
        self.N_datapoint = 20#80 #16
        self.M = 100#600
        self.if_sample_unlearning=False
        self.selected_K_group=[]
        self.sparsity= 0.05

        self.num_cpus = 8 #help="Number of CPUs to assign to a virtual client",
        self.num_gpus = 0.0 #"Ratio of GPU memory to assign to a virtual client"
        self.toy = False
        # "--toy",
        # action="store_true",
        # help="Set to true to use only 10 datasamples for validation. \
        #     Useful for testing purposes. Default: False",
        self.dry = False
        self.client_id=0
        self.use_cuda=False


        self.select_clients=-1
    #     parser.add_argument(
    #     "--dry",
    #     type=bool,
    #     default=False,
    #     required=False,
    #     help="Do a dry-run to check the client",
    # )
    # parser.add_argument(
    #     "--client-id",
    #     type=int,
    #     default=0,
    #     choices=range(0, 10),
    #     required=False,
    #     help="Specifies the artificial data partition of CIFAR10 to be used. \
    #     Picks partition 0 by default",
    # )
    # parser.add_argument(
    #     "--toy",
    #     action="store_true",
    #     help="Set to true to quicky run the client using only 10 datasamples. \
    #     Useful for testing purposes. Default: False",
    # )
    # parser.add_argument(
    #     "--use_cuda",
    #     type=bool,
    #     default=False,
    #     required=False,
    #     help="Set to true to use GPU. Default: False",
    # )
    


def init_global_model(data_name, device):
    # Initialize your model here
    # For demonstration, returning a dummy model
    return f"Model initialized for {data_name} on {device}"

def data_set(data_name):
    # Dummy function to mimic dataset fetching
    return f"Trainset for {data_name}", f"Testset for {data_name}"

def data_init(FL_params):
    # Dummy function to mimic data loader initialization
    return f"All loaders for {FL_params.data_name}", f"Test loaders for {FL_params.data_name}"

def select_clients(FL_params):
    FL_Params.select_clients = np.random.choice(
        range(FL_params.N_total_client),
        size=FL_params.N_client + FL_params.K,
        replace=False
    )
    return FL_Params.select_clients
