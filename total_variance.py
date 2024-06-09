import copy
import torch
import numpy as np
import random
from torch import nn, optim
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
from Fed_Unlearn_base import fedavg, global_train_once, unlearning_step_once, Class_pruner
from FL_base import FL_Finetuned
from sim import simulate_client

# Algorithm 1: Sample-level Unlearning for FATS
def sample_level_unlearning(global_model, old_client_models, client_data_loaders,test_loader, t_u, FL_params, device='cpu'):
    """
    Unlearn a particular sample X_u from the federated learning process.
    """

    client_data_loader_K = list()
    client_models_old_K = list()
    unlearn_global_model = list()
    update_unlearn_global_model=list()
    updated_global_models = list()
    # old_global_models=list()

    for epoch in range(FL_params.global_epoch):
        if (epoch == 0):
            continue
        # print("Federated Unlearning Global Epoch  = {}".format(epoch))
        old_global_model = global_model[epoch]  # oldGM_t
        # old_global_models.append(copy.deepcopy(old_global_model))
    # print('old_global_model', type(old_global_model), len(old_global_models))
    global_model = old_global_model.to(device)
    global_model.train()

    FL_params.K = int((FL_params.rouc * FL_params.local_epoch * FL_params.M) / (
        FL_params.global_epoch
    ))
    FL_params.b = int((FL_params.rous * FL_params.N_datapoint) / (FL_params.rouc * FL_params.local_epoch))

    client_states = {k: {'model': copy.deepcopy(global_model).to(device)} for k in range(FL_params.N_client)}

    split_clients = np.array_split(list(client_states),
                                   np.arange(FL_params.K, len(client_states), FL_params.K))

    flat_clients = [item for sublist in split_clients for item in sublist]
    print('split_clients', split_clients)
    # selected_client = np.random.choice(flat_clients, 1, replace=True)

    selected_client, _ = Class_pruner(global_model, FL_params)
    print('Pruned sample index here:', selected_client)
    FL_params.forget_client_idx = selected_client  # [0]

    for idx in split_clients:
        if (idx == np.array(selected_client)).any():
            FL_params.selected_K_group = idx

    client_models = [copy.deepcopy(global_model) for _ in range(FL_params. N_client)]

    client_K_models=list()
    split_client_models=np.array_split(list(client_models),
                                   np.arange(FL_params.K, len(client_models), FL_params.K))
    for i in FL_params.selected_K_group:
        client_data_loader_K.append(client_data_loaders[i])
        client_models_old_K.append(client_models[i])
    data_loader_ku = client_data_loaders[FL_params.forget_client_idx]

    for client_idx in FL_params.selected_K_group:
        client_K_models.append(client_models[client_idx])#.state_dict())
        print('K client_models', client_K_models)

    for i in client_data_loader_K:
        print('here')
        if data_loader_ku == i:
        # Perform retraining from this point onwards
            client_data_loader = remove_sample_from_loader(data_loader_ku, FL_params.forget_client_idx, FL_params)
            # client_data_loaders.append(client_data_loader)

            FL_params.if_sample_unlearning = True
            fedavg_global_model = fedavg(client_K_models)

            unlearn_global_model.append(fedavg_global_model)

    # print("Updated global", type(updated_global_models), len(updated_global_models))
    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch * FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    # FL_params.global_epoch = CM_intv.shape[0]

    # print('Sample-level Unlearning Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(0, FL_params.K - 1):  # global_epoch):
        if (epoch == 0):
            continue
        print("Sample-level Unlearning Local Calibration Training epoch  = {}".format(epoch))
        global_model = list()

        for e in range(FL_params.global_epoch):
            global_model.append(unlearn_global_model)
            global_model=global_model[-1]

            new_client_models = global_train_once(global_model[0], client_data_loader, test_loader, FL_params)
            unlearning_client_models = new_client_models[-FL_params.K:]
            new_GM = unlearning_step_once(client_models_old_K, unlearning_client_models, fedavg_global_model,
                                          global_model[0])

            unlearn_global_model.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch

    '''fine tuning'''
    print(5 * "#" + "  Federated Fine-tuning Start" + 5 * "#")
    finetuned_global_models = unlearn_global_model
    finetuned_global_model, train_acc, train_epoch = FL_Finetuned(finetuned_global_models[-1], client_data_loaders,
                                                                  client_data_loader[-1], FL_params)
    print("Fine-tuning train acc:%.4f" % train_acc)
    print("Fine-tuning train epoch:%d" % train_epoch)
    print(5 * "#" + "  Federated Fine-tuning End" + 5 * "#")
    updated_global_models.append(finetuned_global_model)

    return updated_global_models

def remove_sample_from_loader(data_loader, sample_to_remove, FL_params):
    """
        从给定的data_loader中移除特定的样本。

        参数:
        - data_loader: 原始的DataLoader。
        - indices_to_remove: 一个包含要移除样本索引的列表。

        返回:
        - 新的不包含指定样本的DataLoader。
        """
    # 获取原始数据集
    original_dataset = data_loader.dataset

    # 计算要保留的样本的索引
    indices_to_keep = [i for i in range(len(original_dataset)) if i != sample_to_remove]

    # 创建一个不包含指定样本的子数据集
    subset_dataset = Subset(original_dataset, indices_to_keep)

    # 使用相同的参数创建一个新的DataLoader，但使用新的子数据集
    new_data_loader = DataLoader(subset_dataset, batch_size=FL_params.b, shuffle=False,
                                 num_workers=data_loader.num_workers)
    new_loaders = list()
    new_loaders.append(new_data_loader)

    return new_loaders

# Algorithm 2: Client-level Unlearning for FATS
def client_level_unlearning(global_model, old_client_models, client_data_loaders,test_loader, FL_params, device='cpu'):
    """
    Unlearn an entire client from the federated learning process.
    t_u: unlearning time step
    k_u: number of target to unlearn
    """
    # Initialize global model parameters
    old_global_models = copy.deepcopy(global_model)
    old_client_models = copy.deepcopy(old_client_models)
    new_GMs = list()

    for epoch in range(FL_params.global_epoch):
        if (epoch == 0):
            continue
        print("Client-level Federated Unlearning Global Epoch  = {}".format(epoch))
        old_global_model = old_global_models[epoch]
        global_model = old_global_model.to(device)
        global_model.train()
        new_GMs.append(global_model)

    FL_params.K = int((FL_params.rouc * FL_params.local_epoch * FL_params.M) / (
        FL_params.global_epoch
    ))
    FL_params.b = int((FL_params.rous * FL_params.N_datapoint) / (FL_params.rouc * FL_params.local_epoch))
    print('K, b: ', FL_params.K, FL_params.b)

    # Time steps
    client_states = {k: {'model': copy.deepcopy(global_model).to(device)} for k in range(FL_params.N_client)}



    split_clients = np.array_split(list(client_states),
                                   np.arange(FL_params.K, len(client_states), FL_params.K))

    # flat_clients = [item for sublist in split_clients for item in sublist]
    print('split_clients', split_clients)

    selected_client, _ = Class_pruner(global_model, FL_params)
    print('selected_client', selected_client)
    print('Pruned sample index here:', selected_client)
    # selected_client = np.random.choice(flat_clients, 1, replace=True)
    for idx in split_clients:
        if (idx == np.array(selected_client)).any():
            FL_params.selected_K_group = idx

    client_models = [copy.deepcopy(global_model) for _ in range(FL_params.N_client)]
    for client_idx in FL_params.selected_K_group:
        client_models[client_idx].load_state_dict(global_model.state_dict())

    FL_params.forget_client_idx = selected_client[-1]

    updated_global_models = list()

    # if 1:#k_u in selected_clients:
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client]
        # print('TEMP HERE',temp)
        temp.pop(FL_params.forget_client_idx)  # During Unlearn, the model saved by the forgotten user pops up
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.N_total_client:]

    new_c_data_loaders = [dl for i, dl in enumerate(client_data_loaders) if i != FL_params.forget_client_idx]
    new_client_data_loaders=[]
    for dl in new_c_data_loaders:
        new_client_data_loader = DataLoader(dl.dataset, batch_size=FL_params.b, shuffle=False)
        new_client_data_loaders.append(new_client_data_loader)

    local_client_models=list()

    for client in FL_params.selected_K_group:
        data_loader = new_client_data_loaders[client]
        local_model = old_client_models[client]
        # Performing local SGD - mini-batch
        for itr in range(FL_params.local_epoch):
            # local_model = local_model1[itr]
            # sampled_dataloader = np.random.choice(list(data_loader), FL_params.b, replace=True)
            for inputs, targets in new_client_data_loaders[itr]:

                optimizer = optim.SGD(local_model.parameters(), lr=FL_params.local_lr)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
            local_client_models.append(local_model)
        # Aggregating local models to update the global model
        # if ii % FL_params.local_epoch == 0:
        unlearn_global_model = list()
        fedavg_global_model = fedavg(local_client_models)
        unlearn_global_model.append(fedavg_global_model)

    updated_global_models.append(unlearn_global_model)


    print("Updated global",updated_global_models,type(updated_global_models),len(updated_global_models),type(updated_global_models[0]))
    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch * FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    # FL_params.global_epoch = CM_intv.shape[0]

    # print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(0, FL_params.K - 1):  # global_epoch):
        if (epoch == 0):
            continue
        print("Client-level Unlearning Local Calibration Training epoch  = {}".format(epoch))
        global_model = updated_global_models[epoch]

        new_client_models = global_train_once(global_model, new_client_data_loaders, test_loader, FL_params)

        new_GM = unlearning_step_once(client_models[epoch], new_client_models, client_models[epoch + 1],
                                      global_model)

        updated_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch

    '''fine tuning'''
    print(5 * "#" + "  Federated Fine-tuning Start" + 5 * "#")
    finetuned_global_models = updated_global_models[-1]
    finetuned_global_model, test_acc, train_epoch,test_loss, train_acc,train_loss = FL_Finetuned(finetuned_global_models[-1], new_client_data_loaders, new_client_data_loader, FL_params)
    print("Fine-tuning train acc:%.4f" % train_acc)
    print("Fine-tuning train epoch:%d" % train_epoch)
    print(5 * "#" + "  Federated Fine-tuning End" + 5 * "#")
    updated_global_models.append(finetuned_global_model)

    return updated_global_models,train_acc,train_loss, test_acc,test_loss

