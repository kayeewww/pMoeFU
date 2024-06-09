import utils
from torch.utils.data import DataLoader
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import datasets
import time
import numpy as np
import sys
from torchvision import datasets, transforms

from total_variance import client_level_unlearning
from model_initiation import model_init
import total_variance
from Fed_Unlearn_base import Class_pruner
from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain, test
from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity
from data_preprocess import data_init, data_init_with_shadow, data_set
from membership_inference import train_attack_model, attack
from model_initiation import Net_cifar10
from config import FL_Params, init_global_model, data_set, data_init

warnings.filterwarnings("ignore")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
FL_Params=FL_Params()
model_cifar10=Net_cifar10()
#trainset, testset, device, init_gm, 1, client_loaders, testloader, FL_Params, select_clients).to_client()
class CifarClient(fl.client.NumPyClient):
    def __init__(
            self,
            trainset: trainset,
            testset: testset,
            device: torch.device,
            model_str: model_cifar10,
            validation_split: int = 0.1,
            client_loaders=None,
            test_loaders=None,
            # FL_Params=None,
            select_clients=FL_Params.forget_client_idx
    ):
        self.select_clients = select_clients
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.model = model_str
        self.client_loaders = client_loaders
        self.test_loader = test_loaders
        # self.FL_Params = FL_Params
        # if model_str == "alexnet":
        #     self.model = utils.load_alexnet(classes=10)
        # else:
        #     self.model = utils.load_efficientnet(classes=10)

    def set_parameters(self, parameters):
        """Loads a alexnet or efficientnet model and replaces it parameters with the
        ones given."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):  # , global_model,FL_Params):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # train_valid = self.trainset.train_test_split(self.validation_split, seed=42)
        # trainset = train_valid["train"]
        # valset = train_valid["test"]

        # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(valset, batch_size=batch_size)

        print(5 * "#" + "  Federated Learning Start" + 5 * "#")
        std_time = time.time()
        old_GMs, old_CMs = FL_Train(self.model, self.client_loaders, testloader, FL_Params)
        # print('Loaders::', self.client_loaders)
        # print('old gms:', old_GMs)
        end_time = time.time()
        time_learn = (end_time-std_time)
        print("#" * 5, "FL_Train time consuming: ", time_learn, "#" * 5)

        print(5 * "#" + "  Federated Learning End" + 5 * "#")

        print('\n')
        """4.2 unlearning  a client，Federated Unlearning"""
        print(5 * "#" + "  Federated Unlearning Start  " + 5 * "#")
        std_time = time.time()
        # Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
        FL_Params.if_unlearning = True
        # unlearn_GMs = unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_Params)

        if (FL_Params.fats_method == 'client'):
            print('#' * 4, 'Client-level Unlearning', '#' * 4)
            self.select_clients = 2
            unlearn_GMs, train_acc, train_loss, test_acc, test_loss = total_variance.client_level_unlearning(old_GMs,
                                                                                                             old_CMs,
                                                                                                             self.client_loaders,
                                                                                                             testloader,
                                                                                                             FL_Params)

        end_time = time.time()
        time_unlearn = (end_time-std_time)
        print("#" * 5, "Total_variance time consuming: ", time_unlearn, "#" * 5)
        print(5 * "#" + "  Federated Unlearning End  " + 5 * "#")

        if (FL_Params.fats_method == 'client'):
            federaser_test_acc, federaser_test_loss = test(unlearn_GMs[-1], testloader)
            # FL_Params.unlearn_client=int(FL_Params.forget_client_idx)
            print('FL_Params.unlearn_client: ', FL_Params.forget_client_idx)
            target_loader = self.client_loaders[FL_Params.forget_client_idx]
            federaser_target_acc, federaser_target_loss = test(unlearn_GMs[-1], target_loader)

        # results = utils.train(self.model, self.client_loaders, self.test_loader, epochs, self.device)
        # results = client_level_unlearning(global_model, self.model, train_loader,val_loader, FL_Params, device='cpu')
        results = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": test_loss,
            "val_accuracy": test_acc,
        }
        print(5 * "*" + "  Result Summary  " + 5 * "*")
        print("[FedEraser] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_test_loss,
                                                                                         federaser_test_acc))

        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.trainset)
        """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

        """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
        print(60 * '=')
        print("Step4. Membership Inference Attack aganist GM...")

        T_epoch = -1
        old_GM = old_GMs[T_epoch]
        attack_model = train_attack_model(old_GM, self.client_loaders, testloader, FL_Params)

        if (FL_Params.fats_method == 'sample'):
            # print('len of sample GM', len(unlearn_GMs))
            target_model = unlearn_GMs  # [T_epoch]
            (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, self.client_loaders, testloader,
                                                FL_Params)

        print("Attacking against FL Unlearn without calibration  ")
        target_model = unlearn_GMs[T_epoch]
        attack(target_model, attack_model, self.client_loaders, testloader, FL_Params)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = test(self.model, testloader)

        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(FL_Params, device: torch.device = "cpu"):
    """Weak tests to check whether all client methods are working as expected."""

    # model = utils.load_efficientnet(classes=10)
    model = Net_cifar10('cpu')
    # trainset, testset = utils.load_partition(0)
    # trainset = trainset.select(range(10))
    # testset = testset.select(range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
        FL_Params
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    # FL_Params = FL_Params()
    torch.manual_seed(FL_Params.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and FL_Params.use_cuda else "cpu"
    )

    init_gm = Net_cifar10()#model_init(FL_Params.data_name, device)

    # if FL_Params.dry:
    #     client_dry_run(FL_Params, device)
    # else:
    # Load a subset of CIFAR-10 to simulate the local data partition
    # trainset, testset = utils.load_partition(args.client_id)

    # if args.toy:
    #     trainset = trainset.select(range(10))
    #     testset = testset.select(range(10))
    # Start Flower client

    # trainset, testset = data_set(FL_Params)
    # transform = transforms.Compose([transforms.ToTensor()])
    # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # 定义数据加载器
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    select_clients = np.random.choice(range(FL_Params.N_client), size=FL_Params.N_client + FL_Params.K, replace=True)
    split_index = [int(trainset.__len__() / FL_Params.N_total_client)] * (FL_Params.N_total_client - 1)
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_Params.N_total_client) * (FL_Params.N_total_client - 1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    print('select_clients: ', select_clients)
    unique_clients = np.unique(select_clients)
    selected_client, _ = Class_pruner(init_gm, FL_Params)
    print('select_clients pruner: ', select_clients)
    # print('split_index', split_index)

    # client_all_loaders, _ = data_init(FL_Params)
    client_all_loaders=[]

    client_loaders = list()
    for ii in range(FL_Params.N_total_client):
        client_all_loaders.append(DataLoader(client_dataset[ii], FL_Params.local_batch_size, shuffle=True))
    # print('Client: client_all_loaders: ', len(client_all_loaders), client_all_loaders[0])
    for idx in select_clients:
        client_loaders.append(client_all_loaders[idx])
        # client_loaders.append(trainloader[idx])
    # print('Client: client_loaders: ', len(client_loaders), client_loaders[0])


    client = CifarClient(trainset, testset, device, init_gm, 1, client_loaders, testloader, unique_clients).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    # port = sys.argv[1]
    main()
