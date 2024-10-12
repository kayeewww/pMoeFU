import zipfile

import torch
import numpy as np
import pandas as pd
import time
import argparse
import os
import copy
import torch.nn.functional as F
import torchtext

from openpyxl.utils.exceptions import InvalidFileException
from torchvision import datasets, transforms
from openpyxl import load_workbook
import torchvision.models as models
import functools
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import resource

# Custom imports
from sample_data import (create_user_dataloaders, remove_class_from_users, print_user_data_distribution,
                         create_user_removedloaders, create_dataloaders_from_indices)

from membership_inference import train_attack_model, attack
# from Fed_moe_base import mix_Train
from torch.utils.tensorboard import SummaryWriter
from sample_data import mnist_noniid2, cifar_noniid2
from FederatedAveraging import FedAvg
from ClientUpdate import ClientUpdate

from class_pruner import calculate_tfidf_scores, select_least_important_clients
from test_model import test_img, test_img_mix
from Models import GateModel, CNNFashion, CNNCifar, Net_adult, Net_purchase


def mix_Train(model_type, opt_in, dataset_train, dataset_test, dict_users, dict_users_val,
              dict_users_test, net_glob_fedAvg, writer, FL_params, vocab, user_train_dataloaders, user_val_dataloaders):
    # training
    val_loss_best = np.inf  # 表示+∞
    counter = 0

    patience = 5
    for n_iter in range(FL_params.global_epoch):  # global epoch
        print('Round {:3d}'.format(n_iter))

        w_fedAvg = []
        alpha = []
        train_loss = []
        val_loss = []
        val_acc = []
        # m = max(int(args.frac * args.num_clients), 1)
        # TObeDO frac=5
        # m = max(int(FL_params.frac), 1)
        # idxs_users = np.random.choice(opt_in, m, replace=False)  # choose opt-in clients
        idxs_users = list(range(len(opt_in)))
        for idx in idxs_users:
            print("FedAvg client %d" % (idx))
            # if FL_params.data_name=='shakespeare' and FL_params.model=='lstm':
            #     clients = []
            #     for i in range(FL_params.N_client):
            #         clients.append(S_ClientUpdate(args=FL_params,  model_type='lstm', train_set=dataset_train, test_set=dataset_test,
            #                                 idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
            #                                 idxs_test=dict_users_test[idx], vocab=vocab))
            # else:
            # print('dict_users', dict_users, 'dict_users_val', dict_users_val)
            client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                                  train_set=dataset_train, test_set=dataset_test,
                                  idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
                                  idxs_test=dict_users_test[idx], user_train_dataloaders=user_train_dataloaders,
                                  user_val_dataloaders=user_val_dataloaders)

            # train FedAvg
            w_glob_fedAvg, train_loss_idx, train_acc_idx = client.train(
                net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device),
                n_epochs=FL_params.local_epoch, learning_rate=5e-5)

            w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))
            train_loss.append(train_loss_idx)
            # Weigh models by client dataset size
            alpha.append(len(dict_users[idx]))

            if (n_iter % 40 == 0):
                val_acc_fed, val_loss_fed = client.validate(net=net_glob_fedAvg, val=True)
                val_acc.append(val_acc_fed)
                val_loss.append(val_loss_fed)

        # update global model weights
        train_loss_avg = sum(train_loss) / len(train_loss)
        writer.add_scalar('fedAvg_train_loss', train_loss_avg, n_iter)

        if (n_iter % 40 == 0):
            val_loss_avg = sum(val_loss) / len(val_loss)
            val_acc_avg = sum(val_acc) / len(val_acc)
            writer.add_scalar('fedAvg_val_loss', val_loss_avg, n_iter)
            writer.add_scalar('fedAvg_val_acc', val_acc_avg, n_iter)
            if (val_loss_avg < val_loss_best):
                print('saving')
                counter = 0
                val_loss_best = val_loss_avg
                w_best_fedavg = w_glob_fedAvg
            else:
                counter = counter + 1

            if (counter == patience):
                break
        print("fedAvg_train_loss, fedAvg_val_loss, fedAvg_val_acc", train_loss_avg, val_loss_avg, val_acc_avg)
        w_glob_fedAvg = FedAvg(w_fedAvg, alpha)
        # copy weight to net_glob
        net_glob_fedAvg.load_state_dict(w_glob_fedAvg)

    net_glob_fedAvg.load_state_dict(w_best_fedavg)
    # torch.save(net_glob_fedAvg.state_dict(), 'multi_global')
    return net_glob_fedAvg, writer


def quick_retrain(model_type, opt_in, dataset_train, dataset_test, dict_users, dict_users_val,
                  dict_users_test, net_glob_fedAvg, writer, FL_params, user_train_dataloaders, user_val_dataloaders):
    # training
    val_loss_best = np.inf  # 表示+∞
    counter = 0
    patience = 5

    for cls in FL_params.unlearn_class:
        target_class_to_remove = cls  # 例如，移除类别 0 的所有样本
        if isinstance(target_class_to_remove, str):
            target_class_to_remove = int(target_class_to_remove.split(" - ")[0])
        else:
            target_class_to_remove = target_class_to_remove  # 已经是整数
        if cls < FL_params.all_classes['first']:
            updated_dict_users, unlearn_dict_users = remove_class_from_users(dict_users, dataset_train,
                                                                             target_class_to_remove)
        else:
            updated_dict_users, unlearn_dict_users = remove_class_from_users(dict_users, dataset_train,
                                                                             target_class_to_remove - 10)
        # print_user_data_distribution(unlearn_dict_users, dataset_train, 5)
        dict_users, dict_users_val, dict_users_test = updated_dict_users, updated_dict_users, updated_dict_users
        for n_iter in range(5):#FL_params.local_epoch):  # global epoch
            # print('RetrainRound {:3d}'.format(n_iter))

            w_fedAvg = []
            alpha = []
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []

            idxs_users = list(range(len(opt_in)-len(FL_params.opt_out)))
            for idx in idxs_users:
                # print("FedAvg client %d" % (idx))
                client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                                      train_set=dataset_train, test_set=dataset_test,
                                      idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
                                      idxs_test=dict_users_test[idx], user_train_dataloaders=user_train_dataloaders,
                                      user_val_dataloaders=user_val_dataloaders)

                # train FedAvg
                w_glob_fedAvg, train_loss_idx, train_acc_idx = client.train(
                    net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device),
                    n_epochs=FL_params.local_epoch, learning_rate=5e-5)

                w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))
                train_loss.append(train_loss_idx)
                train_acc.append(train_acc_idx)
                # Weigh models by client dataset size
                alpha.append(len(dict_users[idx]))

                if (n_iter % 40 == 0):
                    val_acc_fed, val_loss_fed = client.validate(net=net_glob_fedAvg, val=True)
                    val_acc.append(val_acc_fed)
                    val_loss.append(val_loss_fed)

            # update global model weights
            train_loss_avg = sum(train_loss) / len(train_loss)
            train_acc_avg = sum(train_acc) / len(train_acc)
            writer.add_scalar('retrain_fedAvg_train_loss', train_loss_avg, n_iter)

            if (n_iter % 40 == 0):
                val_loss_avg = sum(val_loss) / len(val_loss)
                val_acc_avg = sum(val_acc) / len(val_acc)
                writer.add_scalar('retrain_fedAvg_val_loss', val_loss_avg, n_iter)
                writer.add_scalar('retrain_fedAvg_val_acc', val_acc_avg, n_iter)
                if (val_loss_avg < val_loss_best):
                    print('saving')
                    counter = 0
                    val_loss_best = val_loss_avg
                    w_best_fedavg = w_glob_fedAvg
                else:
                    counter = counter + 1

                if (counter == patience):
                    break
            print("retrain_fedAvg_train_loss, retrain_fedAvg_val_loss, retrain_fedAvg_val_acc", train_loss_avg,
                  val_loss_avg, val_acc_avg)
            w_glob_fedAvg = FedAvg(w_fedAvg, alpha)
            # copy weight to net_glob
            net_glob_fedAvg.load_state_dict(w_glob_fedAvg)

    net_glob_fedAvg.load_state_dict(w_best_fedavg)
    # torch.save(net_glob_fedAvg.state_dict(), 'multi_global')
    return net_glob_fedAvg, train_loss_avg, train_acc_avg, val_loss_avg, val_acc_avg

def read_data(corpus_file, datafields, label_column, doc_start):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        raw_text = []
        for line in f:
            columns = line.strip().split(maxsplit=doc_start)
            doc = columns[-1]
            raw_text.append(doc)
            label = columns[label_column].split(',')[0]
            examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
    return torchtext.data.Dataset(examples, datafields),raw_text
# def get_combined_data(cifar_loaders, fashion_loaders, num_batches=1):
#     cifar_data_list, fashion_data_list = [], []
#
#     cifar_iter = iter(cifar_loaders.values())
#     fashion_iter = iter(fashion_loaders.values())
#
#     for _ in range(num_batches):
#         try:
#             cifar_batch = next(cifar_iter)
#             fashion_batch = next(fashion_iter)
#         except StopIteration:
#             break
#
#         cifar_data, _ = next(iter(cifar_batch))
#         fashion_data, _ = next(iter(fashion_batch))
#         cifar_data_list.append(cifar_data)
#         fashion_data_list.append(fashion_data)
#
#     cifar_combined = torch.cat(cifar_data_list, dim=0)
#     fashion_combined = torch.cat(fashion_data_list, dim=0)
#     return cifar_combined, fashion_combined


# def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
#     """
#
#
#     Parameters
#     ----------
#     old_client_models : list of DNN models
#         When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
#         Models that require forgotten users are not discarded in the Forget function
#     ref_client_models : list of DNN models
#         When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
#         Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model
#
#     global_model_before_forget : The old global model
#         DESCRIPTION.
#     global_model_after_forget : The New global model
#         DESCRIPTION.
#
#     Returns
#     -------
#     return_global_model : After one iteration, the new global model under the forgetting setting
#
#     """
#     # std_time = time.time()
#     old_param_update = dict()  # Model Params： oldCM - oldGM_t
#     new_param_update = dict()  # Model Params： newCM - newGM_t
#
#     new_global_model_state = global_model_after_forget.state_dict()  # newGM_t
#
#     return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
#
#     assert len(old_client_models['local']) == len(new_client_models['local'])
#
#     for layer in global_model_before_forget.state_dict().keys():
#         old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
#         new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
#
#         return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]
#
#         for ii in range(len(new_client_models)):
#             old_param_update[layer] += old_client_models['global'][ii].state_dict()[layer]
#             new_param_update[layer] += new_client_models['global'][ii].state_dict()[layer]
#         old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
#         new_param_update[layer] /= (ii + 1)  # Model Params： newCM
#
#         old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[
#             layer]  # 参数： oldCM - oldGM_t
#         new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[
#             layer]  # 参数： newCM - newGM_t
#
#         step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
#         step_direction = new_param_update[layer] / torch.norm(
#             new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||
#
#         return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction
#
#     return_global_model = copy.deepcopy(global_model_after_forget)
#
#     return_global_model.load_state_dict(return_model_state)
#     # end_time = time.time()
#     # time_learn = (std_time - end_time)
#     # print(" Calibration time consuming = {} secods".format(-time_learn))
#     #
#
#     return return_global_model
# def unlearning_step_once(old_model, new_model, global_model_before_forget, global_model_after_forget):
#     old_param_update = dict()
#     new_param_update = dict()
#     new_global_model_state = global_model_after_forget.state_dict()
#     return_model_state = dict()
#
#     for layer in global_model_before_forget.state_dict().keys():
#         old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
#         new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
#         return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]
#
#         old_param_update[layer] += old_model.state_dict()[layer]
#         new_param_update[layer] += new_model.state_dict()[layer]
#
#         old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]
#         new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]
#
#         step_length = torch.norm(old_param_update[layer])
#         step_direction = new_param_update[layer] / torch.norm(new_param_update[layer])
#
#         return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction
#
#     return_global_model = copy.deepcopy(global_model_after_forget)
#     return_global_model.load_state_dict(return_model_state)
#
#     return return_global_model
def get_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss  # 返回最大内存使用量（以KB为单位）

def train_clients(writer, model_type, idxs_users, dataset_train, dataset_test, dict_users,
                  dict_users_val,
                  dict_users_test, net_glob_fedAvg, net_locals, FL_params, user_train_dataloaders,
                  user_val_dataloaders):
    std_time = time.time()
    finetuned = []
    locals_nets = []
    val_acc_ft, train_acc_ft = [], []
    val_acc_locals, train_acc_locals = [], []
    # n_epoch=1000
    client_list = []

    for idx in idxs_users:
        client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                              train_set=dataset_train, test_set=dataset_test,
                              idxs_train=dict_users[idx], idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx],
                              user_train_dataloaders=user_train_dataloaders, user_val_dataloaders=user_val_dataloaders)
        client_list.append(client)

        # Finetune FedAvg for each client
        print(f"Finetune the client {idx}")
        wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
            net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), n_epochs=FL_params.local_epoch, learning_rate=1e-4,
            val=True)
        val_acc_ft.append(val_acc_finetuned)
        train_acc_ft.append(train_acc_finetuned)

        ft_net = copy.deepcopy(net_glob_fedAvg)
        ft_net.load_state_dict(wt)
        finetuned.append(ft_net)

        # Train local model
        net_local_idx = copy.deepcopy(net_locals).to(FL_params.device)
        w_l, _, val_acc_l, train_acc_l = client.train_finetune(net=ft_net, n_epochs=FL_params.local_epoch,
                                                               learning_rate=1e-4, val=True)
        locals_nets.append(net_local_idx)
        val_acc_locals.append(val_acc_l)
        train_acc_locals.append(train_acc_l)
    end_time = time.time()
    time_learn = (std_time - end_time)
    print(" Local Training time consuming = {} secods".format(-time_learn))
    print(5 * "#" + "  Federated Learning End  " + 5 * "#")
    print(f'Finetune ACC summary: val_acc_ft={val_acc_ft}, train_acc_ft={train_acc_ft}, val_acc_locals={val_acc_locals}'
          f', train_acc_locals={train_acc_locals}')

    return finetuned, locals_nets, val_acc_ft, train_acc_ft, val_acc_locals, train_acc_locals, client_list, time_learn


# def select_clients_based_on_tfidf(tfidf_scores, FL_params):
#     # """
#     # 根据 TF-IDF 分数选择客户端
#     # """
#     # selected_clients = [i for i, score in enumerate(tfidf_scores) if score > threshold]
#     # return selected_clients
#     # 将客户端及其对应的 TF-IDF 分数组合成一个列表
#     client_scores = list(enumerate(tfidf_scores))
#
#     # 按照 TF-IDF 分数升序排序
#     client_scores.sort(key=lambda x: x[1])
#
#     # 选择分数较高的客户端，去掉分数最低的几个
#     selected_clients = [client for client, score in client_scores[FL_params.forget_clients_num:]]
#     FL_params.save_client_idx = selected_clients
#     all_clients = list(range(FL_params.N_client))
#     FL_params.forget_client_idx = [client for client in all_clients if client not in FL_params.save_client_idx]
#
#     return selected_clients


# def mix_train_clients(idxs_users, dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test, net_glob_fedAvg, gate_model, finetuned, net_locals, FL_params):
#
#     std_time=time.time()
#     mix_local, mix_global, mix_gate = [], [], []
#     val_acc_e2e, val_acc_fedavg = [], []
#
#     tfidf_scores = calculate_tfidf_scores(idxs_users, dataset_train, dict_users, net_locals, FL_params)
#     gate_model.update_with_tfidf(tfidf_scores)
#     print("TF-IDF scores used in mix_train_clients:", tfidf_scores)
#
#     mixed_client_models = []
#
#     for i, idx in enumerate(idxs_users):
#         client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=dict_users[idx], idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx])
#
#         print(f"Mix training for client {idx}")
#         # initialize gate with global model conv parameters
#         gate_idx = copy.deepcopy(gate_model).to(FL_params.device)
#
#         if (FL_params.freeze):
#             gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
#             gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
#             ct = 0
#             for child in gate_idx.children():
#                 ct += 1
#                 if (ct < 4):
#                     for param in child.parameters():
#                         param.requiers_grad = False
#             ct = 0
#             # freeze conv layers for finetuned model
#             for child in finetuned[i].children():
#                 ct += 1
#                 if (ct < 4):
#                     for param in child.parameters():
#                         param.requires_grad = False
#         #
#         # gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
#         # gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
#         # for child in list(gate_idx.children())[:3]:
#         #     for param in child.parameters():
#         #         param.requires_grad = False
#         # for child in list(finetuned[i].children())[:3]:
#         #     for param in child.parameters():
#         #         param.requires_grad = False
#
#         gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(
#             net_local=copy.deepcopy(finetuned[i]).to(FL_params.device),
#             net_global=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), gate=gate_idx,
#             train_gate_only=FL_params.train_gate_only, n_epochs=FL_params.local_epoch, learning_rate=1e-4, early_stop=True, val=True)
#
#         net_local=copy.deepcopy(net_locals[idx])
#         gate_idx.load_state_dict(gate_w)
#         mix_gate.append(copy.deepcopy(gate_idx))
#
#         mix_l = copy.deepcopy(net_local)
#         mix_g = copy.deepcopy(net_glob_fedAvg)
#
#         mix_l.load_state_dict(local_w)
#         mix_g.load_state_dict(global_w)
#
#         mix_local.append(mix_l)
#         mix_global.append(mix_g)
#         # mix_gate.append()
#         val_acc_e2e.append(val_acc_e2e_k)
#
#         val_acc_fed, _ = client.validate(net=net_glob_fedAvg.to(FL_params.device), val=True)
#         val_acc_fedavg.append(val_acc_fed)
#
#         # Store client models
#     mixed_client_models.append({
#         'local': mix_local,
#         'global': mix_global,
#         'gate': mix_gate
#     })
#     end_time = time.time()
#     time_learn = (std_time - end_time)
#     print(" Mix Learning time consuming = {} secods".format(-time_learn))
#     print(5 * "#" + "  Federated Learning End" + 5 * "#")
#
#     return mix_local, mix_global, mix_gate, val_acc_e2e, val_acc_fedavg, mixed_client_models

# def resize_and_pad_tensor(tensor, target_shape):
#     """
#     Resize and pad the input tensor to the target shape.
#     Args:
#         tensor (torch.Tensor): Input tensor to be resized and padded.
#         target_shape (tuple): The target shape (C, H, W).
#     Returns:
#         torch.Tensor: Resized and padded tensor.
#     """
#     # Add channel dimension if missing
#     if len(tensor.shape) == 3:
#         tensor = tensor.unsqueeze(1)  # Add channel dimension
#
#     # Resize the tensor to the target height and width
#     tensor_resized = F.interpolate(tensor, size=target_shape[1:], mode='bilinear', align_corners=False)
#
#     # If the number of channels is different, pad the channels
#     if tensor_resized.shape[1] != target_shape[0]:
#         padding = (0, 0, 0, 0, 0, target_shape[0] - tensor_resized.shape[1])
#         tensor_resized = F.pad(tensor_resized, padding, "constant", 0)
#
#     return tensor_resized

#
# def multi_mixtrain(writer, model_type, selected_clients, idxs_users, dataset_train, dataset_test, dict_users,
#                    dict_users_val,
#                    dict_users_test,
#                    net_glob_fedAvg, gate_model, finetuned, net_locals, FL_params, vocab, mix_l):
#     mix_local = []
#     mix_global = []
#     mix_gate = []
#     val_acc_e2e = []
#     val_acc_fedavg = []
#     mixed_client_models = {
#         'local': [],
#         'global': [],
#         'gate': []
#     }
#     tfidf_scores = []
#     tfidf_scores.extend(calculate_tfidf_scores(idxs_users, dataset_train, dict_users, net_locals, FL_params))
#     print('tfidf_scores', tfidf_scores)
#     selected_clients = select_clients_based_on_tfidf(tfidf_scores, FL_params)
#     print('selected_clients11:: ', selected_clients)
#     for i, idx in enumerate(selected_clients):
#         print(f"Mix training for client {idx}")
#         # args, model_type, train_set=None,  test_set=None, idxs_train=None, idxs_val=None, idxs_test=None
#         client = ClientUpdate(args=FL_params,  model_type=model_type, writer=writer,
#                               train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
#                               idxs_test=dict_users_test[idx])
#
#         # initialize gate with global model conv parameters
#         gate_idx = copy.deepcopy(gate_model).to(FL_params.device)
#         # gate_idx2 = copy.deepcopy(gate_model[1]).to(FL_params.device)
#
#         if FL_params.freeze:
#             gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
#             gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
#             ct = 0
#             for child in gate_idx.children():
#                 ct += 1
#                 if ct < 4:
#                     for param in child.parameters():
#                         param.requires_grad = False
#             ct = 0
#             # freeze conv layers for finetuned model
#             for child in finetuned[i].children():
#                 ct += 1
#                 if ct < 4:
#                     for param in child.parameters():
#                         param.requires_grad = False
#
#         gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(
#             net_local=copy.deepcopy(finetuned[i]).to(FL_params.device),
#             net_global=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), gate=gate_idx,
#             train_gate_only=FL_params.train_gate_only, n_epochs=FL_params.local_epoch, learning_rate=1e-4,
#             early_stop=True, val=True, idxs_users=idxs_users, dict_users=dict_users, FL_params=FL_params)
#
#         net_local = copy.deepcopy(net_locals)  # [idx])
#         gate_idx.load_state_dict(gate_w)
#         mix_gate.append(copy.deepcopy(gate_idx))
#
#         mix_l = copy.deepcopy(net_local)
#         mix_g = copy.deepcopy(net_glob_fedAvg)
#
#         mix_l.load_state_dict(local_w)
#         mix_g.load_state_dict(global_w)
#
#         mix_local.append(mix_l)
#         mix_global.append(mix_g)
#         val_acc_e2e.append(val_acc_e2e_k)
#         mixed_client_models['local'].append(copy.deepcopy(mix_l))
#         mixed_client_models['global'].append(copy.deepcopy(mix_g))
#         mixed_client_models['gate'].append(copy.deepcopy(gate_idx))
#
#         val_acc_fed, _ = client.validate(net=net_glob_fedAvg.to(FL_params.device), val=True)
#     val_acc_fedavg.append(val_acc_fed)
#     return selected_clients, mix_local, mix_global, mix_gate, val_acc_e2e, val_acc_fedavg, mixed_client_models
def acculumate_feature(model, loader, stop: int):
    device = next(model.parameters()).device
    model.to(device)
    features = {}
    classes = []
    all_features = []
    all_classes = []

    def hook_func(m, x, y, name, feature_iit):
        f = F.relu(y)
        if f.size()[3] != 0:
            feature = F.avg_pool2d(f, f.size()[3])
            feature = feature.view(f.size()[0], -1)
            feature = feature.transpose(0, 1)

            if name not in feature_iit:
                feature_iit[name] = feature.to(device)
            else:
                feature_iit[name] = torch.cat([feature_iit[name], feature.to(device)], 1)

    hook = functools.partial(hook_func, feature_iit=features)

    handler_list = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)
    # print(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        # inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        # targets = torch.tensor(targets, dtype=torch.long).to(device)
        inputs = inputs.clone().detach().to(device).float()
        targets = targets.clone().detach().to(device).long()

        if batch_idx == stop:
            break
        model.eval()
        classes.extend(targets.cpu().numpy())
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_features.append(outputs)
            all_classes.append(targets)

    all_features = torch.cat(all_features, dim=0)
    all_classes = torch.cat(all_classes, dim=0)
    print(f"Accumulated features shape: {all_features.shape}, classes shape: {all_classes.shape}")

    [k.remove() for k in handler_list]
    # return torch.cat(features), torch.cat(classes)
    # return features, classes
    return all_features, all_classes


def mix_train_clients_step2(writer, tfidf_scores, model_type, idxs_users, dataset_train, dataset_test, dict_users,
                            dict_users_val, dict_users_test,
                            net_glob_fedAvg, gate_model, finetuned, net_locals, FL_params,
                            user_train_dataloaders, user_val_dataloaders):
    mix_local = []
    mix_global = []
    mix_gate = []
    val_acc_e2e = []
    val_acc_fedavg = []
    mixed_client_models = {
        'local': [],
        'global': [],
        'gate': []
    }

    for i, idx in enumerate(idxs_users):
        if i < 5:
            client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                                  train_set=dataset_train[i], test_set=dataset_test[i],
                                  idxs_train=dict_users[0], idxs_val=dict_users_val[0], idxs_test=dict_users_test[0],
                                  user_train_dataloaders=user_train_dataloaders,
                                  user_val_dataloaders=user_val_dataloaders)
        else:
            client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                                  train_set=dataset_train[i], test_set=dataset_test[i],
                                  idxs_train=dict_users[1], idxs_val=dict_users_val[1], idxs_test=dict_users_test[1],
                                  user_train_dataloaders=user_train_dataloaders,
                                  user_val_dataloaders=user_val_dataloaders)

        print(f"Mix training for client {idx}")

        gate_idx = copy.deepcopy(gate_model).to(FL_params.device)

        if FL_params.freeze:
            gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
            gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
            ct = 0
            for child in gate_idx.children():
                ct += 1
                if ct < 4:
                    for param in child.parameters():
                        param.requires_grad = False
            ct = 0
            # freeze conv layers for finetuned model
            for child in finetuned[i].children():
                ct += 1
                if ct < 4:
                    for param in child.parameters():
                        param.requires_grad = False

        # xiaorong_val_acc, _ = client.validate(net=finetuned[i].to(FL_params.device), val=True)
        gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(net_local=copy.deepcopy(finetuned[i]).to(
            FL_params.device),
            net_global=copy.deepcopy(
                net_glob_fedAvg).to(
                FL_params.device), gate=gate_idx,
            train_gate_only=FL_params.train_gate_only,
            n_epochs=FL_params.local_epoch,
            early_stop=True, learning_rate=1e-4,
            val=True, idxs_users=idxs_users,
            dict_users=dict_users, FL_params=FL_params)
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
        # # client.train()
        # w_glob_fedAvg, train_loss_idx, val_acc_fedavg_k = client.train(
        #     net=copy.deepcopy(net_locals[i]).to(FL_params.device),
        #     n_epochs=FL_params.local_epoch, learning_rate=5e-5)
        #
        # val_acc_e2e.append(val_acc_e2e_k)
        # # val_acc_fedavg.append(val_acc_fedavg_k)
        #
        # net_local = copy.deepcopy(net_locals)
        # gate_idx.load_state_dict(gate_w)
        # mix_gate.append(copy.deepcopy(gate_idx))
        #
        # mix_l = copy.deepcopy(net_local)
        # mix_g = copy.deepcopy(net_glob_fedAvg)
        #
        # mix_l.load_state_dict(local_w)
        # mix_g.load_state_dict(global_w)
        #
        # mix_local.append(mix_l)
        # mix_global.append(mix_g)
        mixed_client_models['local'].append(copy.deepcopy(mix_l))
        mixed_client_models['global'].append(copy.deepcopy(mix_g))
        mixed_client_models['gate'].append(copy.deepcopy(gate_idx))

        # val_acc_fed, _ = client.validate(net=net_glob_fedAvg.to(FL_params.device), val=True)
        # val_acc_fedavg.append(val_acc_fed)

    print('val_acc_e2e', val_acc_e2e)
    print('val_acc_fedavg', val_acc_fedavg)

    return mix_local, mix_global, mix_gate, val_acc_e2e, val_acc_fedavg, mixed_client_models


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def mix_train_clients(idxs_users1, idxs_users2, dict_users, dict_users_test, dataset_train, dataset_test, client1,
                      client2, net_glob_fedAvg, gate_model, finetuned, net_locals, FL_params):
    score = []
    score1 = []

    tfidf_dim = 20#FL_params.all_classes['first']+FL_params.all_classes['second']

    num_experts = 10
    # if FL_params.opt_out:


    for local_model in finetuned:
        user_train_dataloaders1 = create_user_dataloaders(dataset_train[0], dict_users[0],
                                                          batch_size=FL_params.local_batch_size)
        user_train_dataloaders2 = create_user_dataloaders(dataset_train[1], dict_users[1],
                                                          batch_size=FL_params.local_batch_size)
        if isinstance(local_model, CNNFashion):
            input_shape2 = (1, 28, 28)
            conv_layers = [(16, 3), (32, 3)]
            linear_layers = [128, 64]
            # tfidf_dim = FL_params.all_classes['first'] + FL_params.all_classes['second']
            model2 = GateModel(input_shape2, conv_layers, linear_layers, tfidf_dim, num_experts,FL_params)
            for loader in user_train_dataloaders2:
                for batch_idx, (images, labels) in enumerate(loader):
                    score1, fusion_model = model2(images, idxs_users2, loader, dict_users[1], local_model,
                                    FL_params)
                    # score1.append(model2.tfidf_scores[0])
            if FL_params.model_1 == FL_params.model_2:
                # for loader in user_train_dataloaders1:
                # print(loader)
                for loader in user_train_dataloaders1:
                    for batch_idx, (images, labels) in enumerate(loader):
                        # features, classes = acculumate_feature(local_model, loader, stop=10)
                        score, fusion_model = model2(images, idxs_users1, loader, dict_users[0], local_model,
                                       FL_params)
                        # score.append(model2.tfidf_scores[0])
            if FL_params.model_1 == 'cnn_fashion' and FL_params.model_2 == 'cnn_mnist':
                for loader in user_train_dataloaders1:
                    for batch_idx, (images, labels) in enumerate(loader):
                        # features, classes = acculumate_feature(local_model, loader, stop=10)
                        score, fusion_model = model2(images, idxs_users1, loader, dict_users[0], local_model,
                                       FL_params)
            new_model_instance2 = model2.generate_new_instance(fusion_model)
        elif isinstance(local_model, models.ResNet):

            input_shape = (3, 32, 32)
            conv_layers = [(16, 3), (32, 3)]
            linear_layers = [128, 64]

            model = GateModel(input_shape, conv_layers, linear_layers, tfidf_dim, num_experts,FL_params)
            for loader in user_train_dataloaders1:
                for batch_idx, (images, labels) in enumerate(loader):
                    score,fusion_model = model(images, idxs_users1, loader, dict_users[0], local_model,
                                  FL_params)

            if FL_params.model_1 == FL_params.model_2:
                for loader in user_train_dataloaders2:
                    for batch_idx, (images, labels) in enumerate(loader):
                        score1,fusion_model = model(images, idxs_users1, loader, dict_users[1], local_model,
                                       FL_params)
            new_model_instance1 = model.generate_new_instance(fusion_model)
        elif isinstance(local_model, CNNCifar):

            input_shape = (3, 32, 32)
            conv_layers = [(16, 3), (32, 3)]
            linear_layers = [128, 64]

            model = GateModel(input_shape, conv_layers, linear_layers, tfidf_dim, num_experts,FL_params)
            for loader in user_train_dataloaders1:
                for batch_idx, (images, labels) in enumerate(loader):
                    score,fusion_model = model(images, idxs_users1, loader, dict_users[0], local_model,
                                  FL_params)

            if FL_params.model_1 == FL_params.model_2:
                for loader in user_train_dataloaders2:
                    for batch_idx, (images, labels) in enumerate(loader):
                        score1,fusion_model = model(images, idxs_users1, loader, dict_users[1], local_model,
                                       FL_params)
            new_model_instance1 = model.generate_new_instance(fusion_model)
        elif isinstance(local_model, Net_purchase):

            input_shape = (600,)  # Purchase 数据集输入形状
            linear_layers = [300, 50]
            tfidf_dim=2
            model = GateModel(input_shape, 0, linear_layers, tfidf_dim, num_experts, FL_params)
            for loader in user_train_dataloaders1:
                for batch_idx, (images, labels) in enumerate(loader):
                    score, fusion_model = model(images, idxs_users1, loader, dict_users[0], local_model,
                                  FL_params)

            if FL_params.model_1 == FL_params.model_2:
                for loader in user_train_dataloaders2:
                    for batch_idx, (images, labels) in enumerate(loader):
                        score1, fusion_model = model(images, idxs_users1, loader, dict_users[1], local_model,
                                       FL_params)
            new_model_instance1 = model.generate_new_instance(fusion_model)

        elif isinstance(local_model, Net_adult):
            input_shape = (108,)  # Adult 数据集输入形状
            linear_layers = [50, 10]
            tfidf_dim=2
            model = GateModel(input_shape, 0, linear_layers, tfidf_dim, num_experts,FL_params)
            for loader in user_train_dataloaders2:
                for batch_idx, (images, labels) in enumerate(loader):
                    score,fusion_model = model(images, idxs_users2, loader, dict_users[1], local_model,
                                  FL_params)

            if FL_params.model_1 == FL_params.model_2:
                for loader in user_train_dataloaders1:
                    for batch_idx, (images, labels) in enumerate(loader):
                        score1,fusion_model = model(images, idxs_users1, loader, dict_users[0], local_model,
                                       FL_params)
            new_model_instance2 = model.generate_new_instance(fusion_model)
    score_list = []
    # values_list = score.tolist()
    values_list = [t.item() for t in score[0]]
    score_list.extend(values_list)
    print('values_list',values_list)
    # score_list.extend(score)
    values_list_matrix = [t.item() for t in score1[0]]
    # values_list_matrix=score1.tolist()
    score_list.extend(values_list_matrix)
    print('values_list_matrix',values_list_matrix)
    # score_list.extend(score1)
    print('score_list', score_list)
    unlearn_class = select_least_important_clients(score_list, num_class_to_select=FL_params.unlearn_class_num)

    if FL_params.opt_out_class:
        unlearn_class.extend(FL_params.opt_out_class)
        print(f'We are going to unlearn opt_out_class:{FL_params.opt_out_class} in unlearn_class')
    print("TFIDF Selected least important class:", unlearn_class)

    #tobedo 0927
    FL_params.unlearn_class = unlearn_class

    user_dataloaders1 = create_user_dataloaders(dataset_train[0], dict_users[0], 64)
    user_dataloaders2 = create_user_dataloaders(dataset_train[1], dict_users[1], 64)
    unlearn_dict_users1 = None
    unlearn_dict_users2 = None
    user_dataloaders1_test = None
    user_dataloaders2_test = None
    unlearn_dataloaders1 = None
    unlearn_dataloaders2 = None
    unlearn_dataloaders1_test = None
    unlearn_dataloaders2_test = None
    updated_dict_users = None
    updated_dict_users1 = None
    updated_test_dict_users2 = None
    updated_test_dict_users1 = None
    unlearn_test_dict_users2 = None
    unlearn_test_dict_users1 = None

    for cls in FL_params.unlearn_class:
        # new_model_instance1.tfidf_scores = new_model_instance1.tfidf_scores.pop(cls)
        class_n1=FL_params.all_classes['first']
        class_n2=FL_params.all_classes['second']
        if cls < FL_params.all_classes['first']:
            class_n1 -=1

            target_class_to_remove = dataset_train[0].classes[cls]
            print(f'We remove the class in {FL_params.data_1}: {target_class_to_remove}')
            FL_params.unlearn_class_tag=target_class_to_remove
            # target_class_to_remove = cls  # 例如，移除类别 0 的所有样本

            updated_dict_users1, unlearn_dict_users1 = remove_class_from_users(dict_users[0], dataset_train[0], cls)
            # updated_class_distribution = get_class_distribution(updated_dict_users, dataset_train[0])
            # print(updated_class_distribution)
            user_dataloaders1, unlearn_dataloaders1 = create_dataloaders_from_indices(dataset_train[0],
                                                                                      updated_dict_users1,
                                                                                      unlearn_dict_users1)
            updated_test_dict_users1, unlearn_test_dict_users1 = remove_class_from_users(dict_users_test[0],
                                                                                       dataset_test[0], cls)

            user_dataloaders1_test, unlearn_dataloaders1_test = create_dataloaders_from_indices(dataset_test[0],
                                                                                                updated_test_dict_users1,
                                                                                                unlearn_test_dict_users1)
            print('Unlearn class in the first type')
            # dict_users[0] = updated_dict_users
            unlearn_dict_users1 = unlearn_dict_users1
            if isinstance(local_model, CNNCifar):
                for user_id, loader in user_dataloaders1.items():
                    for images, labels in loader:
                        new_model_instance1.tfidf_dim = FL_params.all_classes['first'] + FL_params.all_classes['second']
                        score2, update_fusion_model = new_model_instance1(images, idxs_users1, loader, dict_users[1], local_model,
                                                     FL_params)

        else:
            class_n2 -= 1
            target_class_to_remove = dataset_train[1].classes[cls - 10]
            FL_params.unlearn_class_tag = target_class_to_remove
            print(f'We remove the class in {FL_params.data_2}: {target_class_to_remove}')
            updated_dict_users2, unlearn_dict_users2 = remove_class_from_users(dict_users[1], dataset_train[1],
                                                                               cls - 10)
            user_dataloaders2, unlearn_dataloaders2 = create_dataloaders_from_indices(dataset_train[1],
                                                                                      updated_dict_users2,
                                                                                      unlearn_dict_users2)
            updated_test_dict_users2, unlearn_test_dict_users2 = remove_class_from_users(dict_users_test[1],
                                                                                         dataset_test[1], cls - 10)

            user_dataloaders2_test, unlearn_dataloaders2_test = create_dataloaders_from_indices(dataset_test[1],
                                                                                                updated_test_dict_users2,
                                                                                                unlearn_test_dict_users2)
            print('Unlearn class in the second type')  # , len(unlearn_testdata2), len(rest_testdata2))
            # dict_users[1] = updated_dict_users1
            # unlearn_dict_users2 = unlearn_dict_users1

            if isinstance(local_model, CNNFashion):
                for user_id, loader in user_dataloaders2.items():
                    for images, labels in loader:
                        new_model_instance2.tfidf_dim = FL_params.all_classes['first']+FL_params.all_classes['second']
                        # new_model_instance2.tfidf_scores = new_model_instance2.tfidf_scores.pop(cls)
                        score2, update_fusion_model = new_model_instance2(images, idxs_users2, loader, dict_users[1], local_model,
                                                     FL_params)

    return (score_list, updated_dict_users1, unlearn_dict_users1, updated_test_dict_users1, unlearn_test_dict_users1,
            updated_dict_users1, unlearn_dict_users2, updated_test_dict_users2, unlearn_test_dict_users2,
            user_dataloaders1, unlearn_dataloaders1, user_dataloaders1_test, unlearn_dataloaders1_test,
            user_dataloaders2, unlearn_dataloaders2, user_dataloaders2_test, unlearn_dataloaders2_test)



def get_class_distribution(dict_users, dataset):
    """
    获取数据集中每个类别的样本数量。

    Args:
        dict_users (dict): 每个用户的数据集索引字典。
        dataset (Dataset): 数据集。

    Returns:
        dict: 每个类别的样本数量。
    """
    labels = np.array(dataset.targets)
    class_distribution = {}

    for user_id, user_data_indices in dict_users.items():
        user_labels = labels[user_data_indices]
        unique, counts = np.unique(user_labels, return_counts=True)
        class_distribution[user_id] = dict(zip(unique, counts))

    return class_distribution


def generate(dataset, list_classes: list):
    labels = []
    for label_id in list_classes:
        # labels.append(dataset.tensors[1])
        labels.append(list(dataset.classes)[int(label_id)])
    # print(labels)

    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)
    return sub_dataset


def old_tf_cal(dataset_train, dict_users, idxs_users, mix_l, gate_model, FL_params):
    user_train_dataloaders1 = create_user_dataloaders(dataset_train, dict_users,
                                                      batch_size=FL_params.local_batch_size)
    tfidf_scores = calculate_tfidf_scores(idxs_users, user_train_dataloaders1, dict_users[0], mix_l, FL_params)
    weight1 = sum(tfidf_scores) / len(tfidf_scores)
    gate_model.update_with_tfidf(weight1)
    least_important_clients = select_least_important_clients(tfidf_scores,
                                                             num_clients_to_select=FL_params.forget_clients_num)
    FL_params.forget_client_idx = least_important_clients
    print('OLD least_important_clients', least_important_clients)

    return tfidf_scores


def test_clients(writer, tfidf_score, model_type, idxs_users, dataset_train, dict_users, dict_users_val, dataset_test,
                 user_dataloaders1_test, unlearn_dataloaders1_test, unlearn_idxs_test1, dict_users_test,
                 net_glob_fedAvg, locals_nets, finetuned, mix_local, mix_global, mix_gate, user_train_dataloaders,
                 FL_params):
    localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg = [], [], [], []
    len_l = len(idxs_users)
    for i in range(len_l-1):

        # user_train_dataloaders = DataLoader(dataset_train, batch_size=64, shuffle=True)
        # user_val_dataloaders = DataLoader(dataset_train, batch_size=32, shuffle=True)
        # user_test_dataloaders = DataLoader(dataset_test, batch_size=1, shuffle=True)
        if user_dataloaders1_test is None:
            print('original')
            user_dataloaders1_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
        client = ClientUpdate(args=FL_params, model_type=model_type, writer=writer,
                              train_set=dataset_train[i], test_set=dataset_test[i],
                              idxs_train=dict_users[0], idxs_val=dict_users_val[0], idxs_test=dict_users_test[0],
                              user_train_dataloaders=user_train_dataloaders,
                              user_val_dataloaders=user_train_dataloaders,
                              user_test_dataloaders=user_dataloaders1_test)

        localtest_acc_e2e_idx, _ = client.validate_mix(net_l=mix_local[i], net_g=mix_global[i], gate=mix_gate[i],
                                                       val=False, idxs_users=idxs_users, dict_users=dict_users,
                                                       FL_params=FL_params)
        localtest_acc_e2e.append(localtest_acc_e2e_idx)

        localtest_acc_fedavg_idx, _ = client.validate(net=net_glob_fedAvg, val=False)
        localtest_acc_fedavg.append(localtest_acc_fedavg_idx)

        localtest_acc_local_idx, _ = client.validate(net=mix_local[i], val=False)
        localtest_acc_local.append(localtest_acc_local_idx)

        localtest_acc_ft_idx, _ = client.validate(net=finetuned[i], val=False)
        localtest_acc_ft.append(localtest_acc_ft_idx)

    localtest_acc_local_avg = sum(localtest_acc_local) / len(localtest_acc_local)
    localtest_acc_ft_avg = sum(localtest_acc_ft) / len(localtest_acc_ft)
    localtest_acc_e2e_avg = sum(localtest_acc_e2e) / len(localtest_acc_e2e)
    localtest_acc_fedavg_avg = sum(localtest_acc_fedavg) / len(localtest_acc_fedavg)

    test_acc_fedavg, test_acc_local, test_acc_ft, test_acc_e2e = [], [], [], []

    target_acc_fedavg, target_acc_local, target_acc_ft, target_acc_e2e = [], [], [], []
    print("Testing FedAvg...")
    for i, idx in enumerate(idxs_users):
        # print("Testin",user_dataloaders1_test)
        test_acc_fedavg, test_loss_fedavg = test_img(mix_global[i], dataset_test, FL_params, user_dataloaders1_test)
        if unlearn_dataloaders1_test:
            target_acc_fedavg, target_loss_fedavg = test_img(mix_global[i], dataset_test, FL_params,
                                                             unlearn_dataloaders1_test)
            print('target_acc_fedavg, target_loss_fedavg', target_acc_fedavg, target_loss_fedavg)

        # print(f"Testing Locals for client {idx}...")
        test_acc_local_idx, test_loss_local_idx = test_img(mix_local[i], dataset_test, FL_params,
                                                           user_dataloaders1_test)
        test_acc_local.append(test_acc_local_idx)
        if unlearn_dataloaders1_test:
            target_acc_local_idx, target_loss_local_idx = test_img(mix_local[i], dataset_test, FL_params,
                                                                   unlearn_dataloaders1_test)
            target_acc_local.append(target_acc_local_idx)

        # print(f"Testing Finetune for client {idx}...")
        test_acc_ft_idx, test_loss_ft_idx = test_img(finetuned[i], dataset_test, FL_params, user_dataloaders1_test)
        test_acc_ft.append(test_acc_ft_idx)
        if unlearn_dataloaders1_test:
            test_acc_ft_idx, target_loss_ft_idx = test_img(finetuned[i], dataset_test, FL_params, unlearn_dataloaders1_test)
            target_acc_ft.append(test_acc_ft_idx)
        # print(f"Testing mixture for client {idx}...")
        if type(mix_local) == list:
            test_acc_e2e_idx, test_loss_e2e_idx = test_img_mix(mix_local[i], mix_global[i], mix_gate[i], dataset_test,
                                                               FL_params, user_dataloaders1_test)
            if unlearn_dataloaders1_test:
                target_acc_e2e_idx, target_loss_e2e_idx = test_img_mix(mix_local[i], mix_global[i], mix_gate[i],
                                                                       dataset_test,
                                                                       FL_params, unlearn_dataloaders1_test)
        else:
            test_acc_e2e_idx, test_loss_e2e_idx = test_img_mix(mix_local, mix_global[-1], mix_gate[-1], dataset_test,
                                                               FL_params, user_dataloaders1_test)
            if unlearn_dataloaders1_test:
                target_acc_e2e_idx, target_loss_e2e_idx = test_img_mix(mix_local, mix_global[-1], mix_gate[-1],
                                                                       dataset_test,
                                                                       FL_params, unlearn_dataloaders1_test)

        test_acc_e2e.append(test_acc_e2e_idx)
        if unlearn_dataloaders1_test:
            target_acc_e2e.append(target_acc_e2e_idx)

    test_acc_local = sum(test_acc_local) / len(test_acc_local)
    test_acc_ft = sum(test_acc_ft) / len(test_acc_ft)
    test_acc_e2e = sum(test_acc_e2e) / len(test_acc_e2e)
    if unlearn_dataloaders1_test:
        target_acc_local = sum(target_acc_local) / len(target_acc_local)
        target_acc_ft = sum(target_acc_ft) / len(target_acc_ft)
        target_acc_e2e = sum(target_acc_e2e) / len(target_acc_e2e)
    print(f"localtest_acc_local_avg={localtest_acc_local_avg}, localtest_acc_ft_avg={localtest_acc_ft_avg}, "
          f"localtest_acc_e2e_avg={localtest_acc_e2e_avg}, localtest_acc_fedavg_avg={localtest_acc_fedavg_avg}, "
          f"test_acc_local={test_acc_local}, test_acc_ft={test_acc_ft}, test_acc_e2e={test_acc_e2e}, "
          f"test_acc_fedavg={test_acc_fedavg}")
    print(f"target_acc_local={target_acc_local}, target_acc_ft={target_acc_ft}, target_acc_e2e={target_acc_e2e}")
    return localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg, test_acc_local, test_acc_ft, test_acc_e2e, test_acc_fedavg


def save_results(learn_unlearn_mode, forget_clients_num, filename, FL_params, val_acc_avg_e2e, val_acc_avg_locals,
                 val_acc_avg_fedavg,
                 ft_val_acc, ft_train_acc, train_acc_avg_locals, localtest_acc_fedavg,
                 localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, test_acc_fedavg,
                 test_acc_local, test_acc_ft, test_acc_e2e):
    data = {
        'learn_unlearn_mode': learn_unlearn_mode,
        'forget client amount': forget_clients_num,
        'data_name': getattr(FL_params, 'data_name', None),
        'model': getattr(FL_params, 'model', None),
        'global_epoch': getattr(FL_params, 'global_epoch', None),
        'local_epoch': getattr(FL_params, 'local_epoch', None),
        'N_client': getattr(FL_params, 'N_client', None),
        'iid': getattr(FL_params, 'iid', None),
        'p': getattr(FL_params, 'p', None),
        'opt': getattr(FL_params, 'opt', None),
        'n_data': getattr(FL_params, 'n_data', None),
        'frac': getattr(FL_params, 'frac', None),
        'local_lr': getattr(FL_params, 'local_lr', None),
        'train_gate_only': getattr(FL_params, 'train_gate_only', None),
        'val_acc_avg_e2e': val_acc_avg_e2e,
        'val_acc_avg_locals': val_acc_avg_locals,
        'val_acc_avg_fedavg': val_acc_avg_fedavg,
        'ft_val_acc': ft_val_acc,
        'ft_train_acc': ft_train_acc,
        'train_acc_avg_locals': train_acc_avg_locals,
        'localtest_acc_fedavg': localtest_acc_fedavg,
        'localtest_acc_local': localtest_acc_local,
        'localtest_acc_ft': localtest_acc_ft,
        'localtest_acc_e2e': localtest_acc_e2e,
        'test_acc_fedavg': test_acc_fedavg,
        'test_acc_local': test_acc_local,
        'test_acc_ft': test_acc_ft,
        'test_acc_e2e': test_acc_e2e,
        'overlap': getattr(FL_params, 'overlap', None)
    }

    df = pd.DataFrame([data])
    save_dir = 'save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, filename)

    if not os.path.exists(file_path):
        df.to_excel(file_path, index=False)
    else:
        try:
            book = load_workbook(file_path)
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                # writer.sheets = {ws.title: ws for ws in book.worksheets}

                # Ensure at least one sheet is visible
                for sheet in writer.sheets.values():
                    sheet.sheet_state = 'visible'

                start_row = writer.sheets['Sheet1'].max_row
                df.to_excel(writer, startrow=start_row + 1, index=False, header=False)
                writer.save()
        except (InvalidFileException, zipfile.BadZipFile):
            # If the file is invalid, remove it and create a new one
            os.remove(file_path)
            df.to_excel(file_path, index=False)


# def update_clients(writer, model_type, idxs_users, dataset_train, dataset_test, new_dict_users, new_dict_users_val,
#                    new_dict_users_test, net_glob_fedAvg, net_locals, gate_model, FL_params):
#     std_time = time.time()
#     finetuned = []
#     locals_nets = []
#     mix_local = []
#     mix_global = []
#     mix_gate = []
#     # n_epoch=1000
#
#     for idx in idxs_users:
#         client = ClientUpdate(args=FL_params, model_type=model_type, train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=new_dict_users[idx], idxs_val=new_dict_users_val[idx], writer=writer,
#                               idxs_test=new_dict_users_test[idx])
#
#         # Fine-tune the model for the new client
#         fine_tuned_model = copy.deepcopy(net_locals).to(FL_params.device)
#         wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
#             net=fine_tuned_model, n_epochs=FL_params.local_epoch, learning_rate=5e-5, val=True)
#         fine_tuned_model.load_state_dict(wt)
#         finetuned.append(fine_tuned_model)
#
#         # Train local model
#         net_local_idx = copy.deepcopy(net_locals).to(FL_params.device)
#         w_l, _, val_acc_l, train_acc_l = client.train_finetune(net=net_local_idx, n_epochs=FL_params.local_epoch,
#                                                                learning_rate=5e-5, val=True)
#         net_local_idx.load_state_dict(w_l)
#         locals_nets.append(net_local_idx)
#
#         # Mix train clients
#         gate_idx = copy.deepcopy(gate_model).to(FL_params.device)
#         gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
#         gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
#         for child in list(gate_idx.children())[:3]:
#             for param in child.parameters():
#                 param.requires_grad = False
#         for child in list(fine_tuned_model.children())[:3]:
#             for param in child.parameters():
#                 param.requires_grad = False
#
#         gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(
#             net_local=copy.deepcopy(fine_tuned_model).to(FL_params.device),
#             net_global=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), gate=gate_idx,
#             train_gate_only=FL_params.train_gate_only, n_epochs=FL_params.local_epoch, early_stop=True,
#             learning_rate=FL_params.local_lr, val=True, idxs_users=idxs_users, dict_users=new_dict_users, FL_params=FL_params)
#
#         gate_idx.load_state_dict(gate_w)
#         mix_gate.append(copy.deepcopy(gate_idx))
#
#         mix_l = copy.deepcopy(net_locals)
#         mix_g = copy.deepcopy(net_glob_fedAvg)
#
#         mix_l.load_state_dict(local_w)
#         mix_g.load_state_dict(global_w)
#
#         mix_local.append(mix_l)
#         mix_global.append(mix_g)
#     end_time = time.time()
#     time_learn = (std_time - end_time)
#     print(" Updating  time consuming = {} secods".format(-time_learn))
#     print(15 * "#" + "  Updating  End" + 15 * "#")
#
#     return finetuned, locals_nets, mix_local, mix_global, mix_gate


def membership_inference_attack(opt_in, old_client_models, retrain_model, unlearn_GMs, client_loaders, test_loader,
                                class_type, FL_params):
    print("Step5. Membership Inference Attack against GM...")

    # T_epoch = 0  # Using the first (or any specific) epoch for the attack
    old_GM = old_client_models  # ['global']#[0]  # Assuming old_client_models contains the list of old global models
    # if FL_params.multi_model == False:
    #     print("Turn off multi model mode")
    #     old_GM = old_client_models['global'][0]

    attack_model = train_attack_model(old_GM, client_loaders, test_loader, class_type, FL_params)

    print("Attacking against FL Standard...")
    target_model = old_GM
    ACC_old, PRE_old, rec_old, f1_old = attack(target_model, attack_model, client_loaders, test_loader,class_type, FL_params)

    ACC_retrain, PRE_retrain, rec_retrain, f1_retrain=0,0,0,0
    if FL_params.if_retrain:

        if isinstance(retrain_model, type(old_GM)):
            print("Attacking against FL Retrain...")
            target_model = retrain_model
            ACC_retrain, PRE_retrain, rec_retrain, f1_retrain = attack(target_model, attack_model, client_loaders,
                                                                       test_loader, class_type, FL_params)

    print("Attacking against FL Client Unlearn...")
    target_model = unlearn_GMs[0]  # Assuming unlearn_GMs is the updated global model after unlearning
    ACC_unlearn, PRE_unlearn, rec_unlearn, f1_unlearn = attack(target_model, attack_model, client_loaders, test_loader,
                                                               class_type, FL_params)

    return ACC_old, PRE_old, rec_old, f1_old, ACC_retrain if FL_params.if_retrain else None, PRE_retrain if FL_params.if_retrain else None, ACC_unlearn, PRE_unlearn, rec_unlearn, f1_unlearn


# import torch
# import numpy as np
# import pandas as pd
# import time
# import argparse
# import os
# import copy
# from torchvision import datasets, transforms
#
# # ourselves libs
# # from model_initiation import model_init
# from data_preprocess import data_set, dataloader_init, model_init, Net_cifar10, Net_mnist, splitExpertData, splittExpertModel
# from Fed_Unlearn_base import federated_learning_unlearning
# from membership_inference import train_attack_model, attack
# from FL_base import test, FL_Retrain, mix_Train
# import warnings
# from torch.utils.tensorboard import SummaryWriter
# from sample_data import mnist_noniid2, cifar_noniid2
# from FederatedAveraging import FedAvg
# from ClientUpdate import ClientUpdate
# from test_model import test_img, test_img_mix
# from class_pruner import Class_pruner
# from moe import MoE
#
# def mix_fl_finetune(idxs_users,dataset_train,dataset_test,dict_users,dict_users_val,dict_users_test,net_glob_fedAvg,gate_model,net_locals,filename,FL_params):
#     val_acc_locals, val_acc_mix, val_acc_fedavg, val_acc_e2e, val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft, val_acc_e2e_neighbour, val_acc_gateonly = [], [], [], [], [], [], [], [], [], []
#     train_acc_ft, train_acc_locals = [], []
#     acc_test_l, acc_test_m = [], []
#     gate_values = []
#     finetuned = []
#     locals_nets = []
#     # idxs_users = np.random.choice(range(FL_params.N_client), 5, replace=False)  # choose users to evaluate on
#     for idx in idxs_users:
#         client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=dict_users[idx],
#                               idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx])
#
#         # finetune FedAvg for every client
#         print("Finetune each client %d " % (idx))
#         wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
#             net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), n_epochs=500, learning_rate=5e-5, val=True)
#         val_acc_ft.append(val_acc_finetuned)
#         train_acc_ft.append(train_acc_finetuned)
#
#         ft_net = copy.deepcopy(net_glob_fedAvg)
#         ft_net.load_state_dict(wt)
#         finetuned.append(ft_net)
#
#         # train local model
#         # print("Local %d" % (idx))
#         net_local_idx = copy.deepcopy(net_locals).to(FL_params.device)
#         w_l, _, val_acc_l, train_acc_l = client.train_finetune(net=net_local_idx, n_epochs=500, learning_rate=5e-5,
#                                                                val=True)
#
#         net_local_idx.load_state_dict(w_l)
#         locals_nets.append(net_local_idx)
#
#         val_acc_locals.append(val_acc_l)
#         train_acc_locals.append(train_acc_l)
#     mix_local = []
#     mix_global = []
#     mix_gate = []
#     for i, idx in enumerate(idxs_users):
#         client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=dict_users[idx],
#                               idxs_val=dict_users_val[idx], idxs_test=dict_users_test[idx])
#
#         print("Client Update: E2e %d" % (idx))
#         gate_idx = copy.deepcopy(gate_model).to(FL_params.device)
#         # initialize gate with global model conv parameters
#         # if (args.freeze):
#         if (1):
#             gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
#             gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
#             ct = 0
#             for child in gate_idx.children():
#                 ct += 1
#                 if (ct < 4):
#                     for param in child.parameters():
#                         param.requiers_grad = False
#             ct = 0
#             # freeze conv layers for finetuned model
#             for child in finetuned[i].children():
#                 ct += 1
#                 if (ct < 4):
#                     for param in child.parameters():
#                         param.requires_grad = False
#
#         gate_w, local_w, global_w, _, val_acc_e2e_k = client.train_mix(
#             net_local=copy.deepcopy(finetuned[i]).to(FL_params.device),
#             net_global=copy.deepcopy(net_glob_fedAvg).to(FL_params.device), gate=gate_idx,
#             train_gate_only=FL_params.train_gate_only, n_epochs=500, early_stop=True, learning_rate=FL_params.local_lr,
#             val=True)
#
#         gate_idx.load_state_dict(gate_w)
#         mix_gate.append(copy.deepcopy(gate_idx))
#
#         mix_l = copy.deepcopy(net_locals)
#         mix_g = copy.deepcopy(net_glob_fedAvg)
#
#         mix_l.load_state_dict(local_w)
#         mix_g.load_state_dict(global_w)
#
#         mix_local.append(mix_l)
#         mix_global.append(mix_g)
#
#         val_acc_e2e.append(val_acc_e2e_k)
#
#         # evaluate FedAvg on local dataset
#         val_acc_fed, _ = client.validate(net=net_glob_fedAvg.to(FL_params.device), val=True)
#         val_acc_fedavg.append(val_acc_fed)
#         # evaluate with local client test set
#     localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg = [], [], [], []
#     for i, idx in enumerate(idxs_users):
#         client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
#                               idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
#                               idxs_test=dict_users_test[idx])
#
#         localtest_acc_e2e_idx, _ = client.validate_mix(net_l=mix_local[i],
#                                                        net_g=mix_global[i],
#                                                        gate=mix_gate[i], val=False)
#
#         localtest_acc_e2e.append(localtest_acc_e2e_idx)
#
#         localtest_acc_fedavg_idx, _ = client.validate(net=net_glob_fedAvg, val=False)
#         localtest_acc_fedavg.append(localtest_acc_fedavg_idx)
#
#         localtest_acc_local_idx, _ = client.validate(net=locals_nets[i], val=False)
#         localtest_acc_local.append(localtest_acc_local_idx)
#
#         localtest_acc_ft_idx, _ = client.validate(net=finetuned[i], val=False)
#         localtest_acc_ft.append(localtest_acc_ft_idx)
#     localtest_acc_local = sum(localtest_acc_local) / len(localtest_acc_local)
#     localtest_acc_ft = sum(localtest_acc_ft) / len(localtest_acc_ft)
#     localtest_acc_e2e = sum(localtest_acc_e2e) / len(localtest_acc_e2e)
#     localtest_acc_fedavg = sum(localtest_acc_fedavg) / len(localtest_acc_fedavg)
#
#     # evaluate all models on balanced (global) dataset
#     test_acc_local, test_acc_ft, test_acc_e2e = [], [], []
#     print("testing FedAvg...")
#     test_acc_fedavg, _ = test_img(net_glob_fedAvg, dataset_test, FL_params)
#     for i, idx in enumerate(idxs_users):
#         print(idx)
#         print("testing Locals...")
#         test_acc_local_idx, _ = test_img(locals_nets[i], dataset_test, FL_params)
#         print("testing Finetune...")
#         test_acc_ft_idx, _ = test_img(finetuned[i], dataset_test, FL_params)
#         print("testing mixture...")
#         test_acc_e2e_idx, _ = test_img_mix(mix_local[i], mix_global[i], mix_gate[i], dataset_test, FL_params)
#
#         test_acc_local.append(test_acc_local_idx)
#         test_acc_ft.append(test_acc_ft_idx)
#         test_acc_e2e.append(test_acc_e2e_idx)
#
#     test_acc_local = sum(test_acc_local) / len(test_acc_local)
#     test_acc_ft = sum(test_acc_ft) / len(test_acc_ft)
#     test_acc_e2e = sum(test_acc_e2e) / len(test_acc_e2e)
#
#     # Calculate validation and test accuracies
#
#     val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)
#     # val_acc_avg_locals = np.nan
#
#     train_acc_avg_locals = sum(train_acc_locals) / len(train_acc_locals)
#     # train_acc_avg_locals = np.nan
#
#     val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
#     # val_acc_avg_e2e = np.nan
#
#     # val_acc_avg_e2e_neighbour = sum(val_acc_e2e_neighbour) / len(val_acc_e2e_neighbour)
#     val_acc_avg_e2e_neighbour = np.nan
#
#     # val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
#     val_acc_avg_3 = np.nan
#
#     # val_acc_avg_gateonly = sum(val_acc_gateonly) / len(val_acc_gateonly)
#     val_acc_avg_gateonly = np.nan
#
#     # val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
#     val_acc_avg_rep = np.nan
#
#     # val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
#     val_acc_avg_repft = np.nan
#
#     val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)
#
#     ft_val_acc = sum(val_acc_ft) / len(val_acc_ft)
#     # ft_val_acc = np.nan
#
#     ft_train_acc = sum(train_acc_ft) / len(train_acc_ft)
#     # ft_train_acc = np.nan
#     # 创建字典，其中键是列名，值是相应的数据
#     data = {
#         'data_name': [FL_params.data_name],
#         'model': [FL_params.model],
#         'global_epoch': [FL_params.global_epoch],
#         'local_epoch': [FL_params.local_epoch],
#         'N_client': [FL_params.N_client],
#         'iid': [FL_params.iid],
#         'p': [FL_params.p],
#         'opt': [FL_params.opt],
#         'n_data': [FL_params.n_data],
#         'frac': [FL_params.frac],
#         'local_lr': [FL_params.local_lr],
#         'train_gate_only': [FL_params.train_gate_only],
#         'val_acc_avg_e2e': [val_acc_avg_e2e],
#         'val_acc_avg_e2e_neighbour': [val_acc_avg_e2e_neighbour],
#         'val_acc_avg_locals': [val_acc_avg_locals],
#         'val_acc_avg_fedavg': [val_acc_avg_fedavg],
#         'ft_val_acc': [ft_val_acc],
#         'val_acc_avg_3': [val_acc_avg_3],
#         'val_acc_avg_rep': [val_acc_avg_rep],
#         'val_acc_avg_repft': [val_acc_avg_repft],
#         'ft_train_acc': [ft_train_acc],
#         'train_acc_avg_locals': [train_acc_avg_locals],
#         'val_acc_avg_gateonly': [val_acc_avg_gateonly],
#         'localtest_acc_fedavg': [localtest_acc_fedavg],
#         'localtest_acc_local': [localtest_acc_local],
#         'localtest_acc_ft': [localtest_acc_ft],
#         'localtest_acc_e2e': [localtest_acc_e2e],
#         'test_acc_fedavg': [test_acc_fedavg],
#         'test_acc_local': [test_acc_local],
#         'test_acc_ft': [test_acc_ft],
#         'test_acc_e2e': [test_acc_e2e],
#         'overlap': [FL_params.overlap]
#     }
#
#     # 创建DataFrame
#     df = pd.DataFrame(data)
#
#     # 将DataFrame写入Excel文件
#     filename = 'output.xlsx'
#     df.to_excel('save/' + filename, index=False)
#
#
#     with open('save/' + filename, 'a') as f1:
#         f1.write(
#             f'{FL_params.data_name};{FL_params.model};{FL_params.global_epoch};{FL_params.local_epoch};{FL_params.N_client};{FL_params.iid};{FL_params.p};{FL_params.opt};{FL_params.n_data};{FL_params.frac};{FL_params.local_lr};{FL_params.train_gate_only};{val_acc_avg_e2e};{val_acc_avg_e2e_neighbour};{val_acc_avg_locals};{val_acc_avg_fedavg};{ft_val_acc};{val_acc_avg_3};{val_acc_avg_rep};{val_acc_avg_repft};{ft_train_acc};{train_acc_avg_locals};{val_acc_avg_gateonly};{localtest_acc_fedavg};{localtest_acc_local};{localtest_acc_ft};{localtest_acc_e2e};{test_acc_fedavg};{test_acc_local};{test_acc_ft};{test_acc_e2e};{FL_params.overlap}')
#         f1.write("\n")
#
#     return net_glob_fedAvg, locals_nets
def is_resnet18(model):
    return isinstance(model, models.ResNet) and model.layer4[1].conv1.out_channels == 512
