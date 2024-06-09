from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from datasets import Dataset

import flwr as fl
import torch

import utils
from flwr.common import Metrics
from flwr.common.typing import Scalar

import warnings

from flwr_datasets import FederatedDataset
from FL_base import test, fedavg
from model_initiation import Net_cifar10, model_init
from data_preprocess import data_init, data_init_with_shadow, data_set
from config import FL_Params, init_global_model, data_set, data_init


warnings.filterwarnings("ignore")
    

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}
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

def get_evaluate_fn(val_loader, model: torch.nn.Module, toy: bool,FL_params):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    # centralized_data = utils.load_centralized_data()
    centralized_data = utils.load_centralized_data()
    # val_loader = DataLoader(centralized_data, batch_size=16)

    _, centralized_data = data_set(FL_params.data_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # if toy:
    #     # use only 10 samples as validation set
    #     centralized_data = centralized_data.select(range(10))
    # _, val_loader = data_init(FL_params)
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


    # val_loader1 = DataLoader(centralized_data, batch_size=16)
    trainset, testset = data_set(FL_params.data_name)
    # data, label = testset[0]
    # print("First dataset item:", testset[1])
    # print("Type of first item:", type(testset[1]))
    client_all_loaders, _ = data_init(FL_params)
    # print('Test loader:  ', test_loader)

    # print("Data type:", type(data), "Label type:", type(label))
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # 定义数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # print('Batch', FL_params.test_batch_size)
    # val_loader1 = torch.utils.data.DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
    # print('Val_loader:  ', val_loader1)
    # for batch in test_loader:
    #     print(type(batch))
    #     print(len(batch))
    #     print(batch)
    #     break

    # 初始化一个空列表来存储结果
    # test_data = []

    # 遍历数据加载器，并将 (batch_idx, (inputs, targets)) 添加到列表中
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    #     test_data.append((batch_idx, (inputs, targets)))
    


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        # state_dict = OrderedDict()
        
        # for k, v in params_dict:
        #     # Ensure the tensors are on the same device and type as the model's parameters
        #     # param_device = model.state_dict()[k].device
        #     # state_dict[k] = torch.tensor(v, device=params_dict[k].device, dtype=params_dict[k].dtype)
        #     params_dict = zip(model.state_dict().keys(), parameters)
        #
        #     # Convert params_dict from zip to a dictionary
        #     params_dict = {k: torch.tensor(v, device=model.state_dict()[k].device, dtype=model.state_dict()[k].dtype)
        #                    for k, v in params_dict}
        #
        #     # Update model's state_dict
        #     model.load_state_dict(params_dict, strict=True)

        # for k, v in params_dict:
        #     # Ensure the tensors are on the same device and type as the model's parameters
        #     tensor = torch.tensor(v, device=model.state_dict()[k].device, dtype=model.state_dict()[k].dtype)
        #     state_dict[k] = tensor
        # model.load_state_dict(state_dict, strict=True)

        # epoch=0
        # global_model = fedavg(model)
        # print("Global Federated Learning epoch = {}".format(epoch))
        #
        # (val_acc, test_loss) = test(global_model, val_loader1)


        accuracy, loss = test(model, test_loader)
        
        print(f"Server Round {server_round}: Loss = {loss}, Accuracy = {accuracy}")
        return loss, {"Server test accuracy": accuracy}

    return evaluate


def aggregate_metrics(fit_metrics):
    aggregated_metrics = {}

    for metrics in fit_metrics:
        if isinstance(metrics, tuple) and len(metrics) > 0 and isinstance(metrics[0], dict):
            metrics_dict = metrics[0]
            for key, value in metrics_dict.items():
                if key in aggregated_metrics:
                    aggregated_metrics[key].append(value)
                else:
                    aggregated_metrics[key] = [value]
        else:
            print("Unexpected data format:", metrics)

    # Average out the metrics
    for key in aggregated_metrics:
        aggregated_metrics[key] = sum(aggregated_metrics[key]) / len(aggregated_metrics[key])
    
    return aggregated_metrics
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    FL_params = FL_Params()
    torch.manual_seed(FL_params.seed)

    #kwargs for data loader
    print(60*'=')
    print("Step1. Federated Learning Settings \n We use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment.\n"))


    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    #加载数据
    #Net_cifar10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_global_model = model_init(FL_params.data_name, device)
    # 100， 1 DataLoader
    client_all_loaders, test_loader = data_init(FL_params)
    first_batch = next(iter(test_loader))
    print("Server.py: First batch:", first_batch)
    # print("First batch data type:", type(first_batch[0]), "First batch label type:", type(first_batch[1]))

    # model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    model_parameters = [val.cpu().numpy() for _, val in init_global_model.state_dict().items()]
    # 10 indexes
    FL_Params.select_clients=np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client+FL_params.K, replace=False)
    # selected_clients = np.random.choice(FL_params.N_total_client, size=int(FL_params.N_client))
    # client_loaders = list()
    # for idx in select_clients:
    #     client_loaders.append(client_all_loaders[idx])

    # trainset, testset = data_set(FL_params.data_name)
    # print('len of selected client loader,', len(client_loaders))
    # evaluate_fn=get_evaluate_fn(init_global_model, FL_params.toy,FL_params),
        
    
    # Create strategy ----Method----federated_learning_unlearning
    # cifar10_fds, centralized_testset=get_fds_and_centralized_tesetset(FL_params)
    
    # model = utils.load_Net_cifar10()#.state_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net_cifar10(device=device)

    print('Server.py: model typr: ', type(model), model)
    # cifar10_fds = FederatedDataset(dataset="cifar10", partitioners={"train": FL_params.N_client})
    # centralized_testset = cifar10_fds.load_split("test")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,#1.0,
        fraction_evaluate=0.05,#1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(test_loader, model, FL_params.toy, FL_params),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        # fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.1,  # Sample 10% of available clients for training
    #     fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
    #     min_fit_clients=10,
    #     min_evaluate_clients=10,
    #     min_available_clients=10,
    #     #TODO client
    #     on_fit_config_fn=config_for_client_002,
    #     #(client_id: str, client_resources: Dict[str, Dict[str, int]]) -> Dict[str, int]
    #     evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
    #     evaluate_fn=get_evaluate_fn(centralized_testset, client_model, FL_params),  # Global evaluation function
    #     # initial_parameters=fl.common.ndarrays_to_parameters(params),
    # )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:8080",#"10.3.242.104:22",#
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
