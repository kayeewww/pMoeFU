import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor, Normalize, Compose
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
# from model_initiation import Net_mnist,Net_cifar10,Net_adult,Net_purchase
from FL_base import test, global_train_once
from data_preprocess import data_set
from flwr_datasets import FederatedDataset
import ray
# from utils import Net, train, test, apply_transforms
# class Net(nn.Module):
#     def __init__(self, num_classes: int = 10) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
#
# parser.add_argument(
#     "--num_cpus",
#     type=int,
#     default=1,
#     help="Number of CPUs to assign to a virtual client",
# )
# parser.add_argument(
#     "--num_gpus",
#     type=float,
#     default=0.0,
#     help="Ratio of GPU memory to assign to a virtual client",
# )

# NUM_CLIENTS = 100
# NUM_ROUNDS = 10


# Flower client, adapted from Pytorch quickstart example
# class FL_params:
#     pass

import logging
import os

# log_directory = os.path.expanduser('~/log/ray')  # This will create the log directory in the home folder
# os.makedirs(log_directory, exist_ok=True)

# log_file_path = os.path.join(log_directory, 'simulate.log')


# # Configure logging
# logging.basicConfig(level=logging.DEBUG,  # Capture all levels of messages
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     filename=log_file_path,  # Output to a file
#                     filemode='w')  # Overwrite the file with each run


# @ray.remote
# def raylog_function():
#     logging.info("This function is running remotely")
#     logging.debug("This is a debug message")

# raylog_function.remote()



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, client_model, FL_params, **kwargs):
        self.trainset = trainset
        self.valset = valset
        self.FL_params = FL_params

        # Instantiate model
        self.model = client_model#Net_cifar10()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=self.FL_params.test_batch_size, shuffle=True)
        testloader = DataLoader(self.valset, batch_size=self.FL_params.test_batch_size)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        trainloaders=[]
        for client in range(self.FL_params.N_client):
            trainloaders.append(trainloader)
        # print(trainloaders)
        client_models=global_train_once(self.model, trainloaders, test_loader=testloader, FL_params=self.FL_params)

        self.model=client_models[-1]
        # print(self.model)
        # (self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=self.FL_params.test_batch_size)

        # Evaluate
        loss, accuracy = test(self.model, valloader)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def get_client_fn(dataset, client_model, FL_params):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        '''load data and model'''
        if FL_params.data_name == 'cifar10':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # net = model_init('cifar10', device)
            trainset, testset = data_set('cifar10')
            total_classes = 10  # [0-9]
            cifar10_fds = FederatedDataset(dataset="cifar10", partitioners={"train": FL_params.N_client})
            centralized_testset = cifar10_fds.load_split("test")

        elif FL_params.data_name == 'mnist':
            trainset, testset = data_set('mnist')
            # net = model_init('mnist', FL_params.model_name)
            total_classes = 10  # [0-9]
        elif FL_params.data_name == 'cifar100':
            trainset, testset = data_set('cifar100')
            # net = model_init('cifar100', FL_params.model_name)
            total_classes = 10  # [0-9]
        elif FL_params.data_name == 'adult':
            trainset, testset = data_set('adult')
            # net = model_init('adult', FL_params.model_name)
            total_classes = 2  # [0-1]
        elif FL_params.data_name == 'purchase':
            trainset, testset = data_set('purchase')
            # net = model_init('purchase', FL_params.model_name)
            total_classes = 2  # [0-9]

        # # # Let's get the partition corresponding to the i-th client
        # client_dataset = dataset.load_partition(int(cid), "train")
        #
        # # Now let's split it into train (90%) and validation (10%)
        # client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)
        #
        # trainset = client_dataset_splits["train"]
        # valset = client_dataset_splits["test"]
        #
        # # Now we apply the transform to each batch.
        # trainset = trainset.with_transform(apply_transforms)
        # testset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset, centralized_testset, client_model, FL_params).to_client()
    #self, trainset, valset, client_model, FL_params, **kwargs

    return client_fn
def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

# def fit_config(server_round: int) -> Dict[str, Scalar]:
#     """Return a configuration with static batch size and (local) epochs."""
#     config = {
#         "epochs": 1,  # Number of local epochs done by clients
#         # "batch_size": 32,  # Batch size to use by clients during fit()
#         "batch_size": 16#32
#     }
#     return config

def fit_config(client_id: str, client_resources: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """
    Return a configuration with customized batch size and local epochs based on client resources.

    Parameters:
        client_id (str): Unique identifier for the client.
        client_resources (Dict[str, Dict[str, int]]): A dictionary containing resource specifications for each client.
    
    Returns:
        Dict[str, int]: Configuration dictionary with epochs and batch_size tailored to the client's resources.
    """

    # Default configuration if client specifics are not found
    default_config = {
        "epochs": 1,
        "batch_size": 16
    }

    # Retrieve specific configuration for the given client_id, if available
    client_config = client_resources.get(client_id, default_config)

    # Set epochs and batch size based on client resources
    config = {
        "epochs": client_config.get("epochs", 1),  # Defaults to 1 if not specified
        "batch_size": client_config.get("batch_size", 16)  # Defaults to 16 if not specified
    }
    
    return config

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(
    centralized_testset: Dataset, client_model, FL_params
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = client_model#Net_cifar10()
        set_params(model, parameters)
        model.to(device)

        '''load data and model'''
        if FL_params.data_name == 'cifar10':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # net = model_init('cifar10', device)
            trainset, testset = data_set('cifar10')
            total_classes = 10  # [0-9]
        elif FL_params.data_name == 'mnist':
            trainset, testset = data_set('mnist')
            # net = model_init('mnist', FL_params.model_name)
            total_classes = 10  # [0-9]
        elif FL_params.data_name == 'cifar100':
            trainset, testset = data_set('cifar100')
            # net = model_init('cifar100', FL_params.model_name)
            total_classes = 10  # [0-9]
        elif FL_params.data_name == 'adult':
            trainset, testset = data_set('adult')
            # net = model_init('adult', FL_params.model_name)
            total_classes = 2  # [0-1]
        elif FL_params.data_name == 'purchase':
            trainset, testset = data_set('purchase')
            # net = model_init('purchase', FL_params.model_name)
            total_classes = 2  # [0-9]

        # Apply transform to dataset
        # testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()
        #TODO 原来是这样的诶
        testloader = DataLoader(testset, batch_size=FL_params.test_batch_size)
        loss, accuracy = test(model, testloader)

        return loss, {"accuracy": accuracy}

    return evaluate



def get_fds_and_centralized_tesetset(FL_params):
    # Download MNIST dataset and partition it
    # mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    # centralized_testset = mnist_fds.load_split("test")
    '''load data and model'''
    if FL_params.data_name == 'cifar10':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net = model_init('cifar10', device)
        trainset, testset = data_set('cifar10')
        total_classes = 10  # [0-9]
    elif FL_params.data_name == 'mnist':
        trainset, testset = data_set('mnist')
        # net = model_init('mnist', FL_params.model_name)
        total_classes = 10  # [0-9]
    elif FL_params.data_name == 'cifar100':
        trainset, testset = data_set('cifar100')
        # net = model_init('cifar100', FL_params.model_name)
        total_classes = 10  # [0-9]
    elif FL_params.data_name == 'adult':
        trainset, testset = data_set('adult')
        # net = model_init('adult', FL_params.model_name)
        total_classes = 2  # [0-1]
    elif FL_params.data_name == 'purchase':
        trainset, testset = data_set('purchase')
        # net = model_init('purchase', FL_params.model_name)
        total_classes = 2  # [0-9]

    cifar10_fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
    centralized_testset = cifar10_fds.load_split("test")
    return trainset, testset #cifar10_fds, centralized_testset



def config_client_and_server_fl(client_model,FL_params):
    cifar10_fds, centralized_testset=get_fds_and_centralized_tesetset(FL_params)
    # Example dictionary containing resource configurations for different clients
    client_resources = {
        "client_001": {"epochs": 1, "batch_size": 32},
        "client_002": {"epochs": 1, "batch_size": 8},  # Simulate lower capability device
        "client_003": {"epochs": 1, "batch_size": 24}
    }

    # Getting configuration for client_002
    config_for_client_002 = fit_config("client_002", client_resources)
    print("Configuration for Client 002:", config_for_client_002)

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        #TODO client
        on_fit_config_fn=config_for_client_002,
        #(client_id: str, client_resources: Dict[str, Dict[str, int]]) -> Dict[str, int]
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset, client_model, FL_params),  # Global evaluation function
        # initial_parameters=fl.common.ndarrays_to_parameters(params),
    )
    # client_list=[]
    # ClientApp for Flower-Next
    client = fl.client.ClientApp(
        client_fn=get_client_fn(cifar10_fds, client_model, FL_params),
    )

    # ServerApp for Flower-Next
    server = fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=10),#FL_params.global_epoch/2),
        strategy=strategy,
    )
    return strategy, client, server


def simulate_client(client_model, FL_params):
    # Initialize Ray with a custom temporary directory
    ray.init(
        logging_level='DEBUG',  # Set the desired level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        logging_dir='/log/ray',  # Optional: Specify a directory for Ray logs
        log_to_driver=True,  # Logs from workers will be sent to the driver
        _temp_dir="/session"
        )
    
    cifar10_fds, centralized_testset = get_fds_and_centralized_tesetset(FL_params)
    strategy, client, server = config_client_and_server_fl(client_model, FL_params)

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": FL_params.num_cpus,
        "num_gpus": FL_params.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(cifar10_fds, client_model, FL_params),
        num_clients=FL_params.N_total_client,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=FL_params.global_epoch),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


# if __name__ == "__main__":
#     FL_params = Arguments()
#     main(FL_params)
