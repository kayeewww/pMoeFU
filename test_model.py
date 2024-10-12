#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
def test_img(net_g, datatest, args, test_dataloader):
    with torch.no_grad():
        net_g.eval()
        total_test_loss = 0
        total_correct_g = 0
        total_samples = 0

        if isinstance(test_dataloader, DataLoader):
            user_total_samples = 0  # Initialize inside loop for dataloader
            user_test_loss = 0
            user_correct_g = 0
            # total_samples = len(test_dataloader.dataset)
            for data, target in test_dataloader:

                data, target = data.to(args.device), target.to(args.device)
                user_total_samples += data.size(0)
                log_probs_g = net_g(data.float())
                loss = nn.CrossEntropyLoss()(log_probs_g, target)
                user_test_loss += loss.item()

                y_pred_g = log_probs_g.data.max(1, keepdim=True)[1]
                user_correct_g += y_pred_g.eq(target.data.view_as(y_pred_g)).long().cpu().sum().item()

            total_test_loss += user_test_loss
            total_correct_g += user_correct_g
            total_samples += user_total_samples

        elif isinstance(test_dataloader, dict):
            for user_id, loader in test_dataloader.items():
                # user_total_samples = len(loader.dataset)
                user_total_samples = 0
                user_test_loss = 0
                user_correct_g = 0
                for data, target in loader:
                    data, target = data.to(args.device), target.to(args.device)
                    user_total_samples += data.size(0)
                    log_probs_g = net_g(data.float())
                    loss = nn.CrossEntropyLoss()(log_probs_g, target)
                    user_test_loss += loss.item()

                    y_pred_g = log_probs_g.data.max(1, keepdim=True)[1]
                    user_correct_g += y_pred_g.eq(target.data.view_as(y_pred_g)).long().cpu().sum().item()

                total_test_loss += user_test_loss
                total_correct_g += user_correct_g
                total_samples += user_total_samples
        elif isinstance(test_dataloader, list):

            for loader in test_dataloader:
                user_total_samples = 0  # Initialize inside loop for loaders
                user_test_loss = 0
                user_correct_g = 0
                # total_samples = len(loader.dataset)
                for data, target in loader:

                    data, target = data.to(args.device), target.to(args.device)
                    user_total_samples += data.size(0)
                    log_probs_g = net_g(data.float())
                    loss = nn.CrossEntropyLoss()(log_probs_g, target)
                    user_test_loss += loss.item()

                    y_pred_g = log_probs_g.data.max(1, keepdim=True)[1]
                    user_correct_g += y_pred_g.eq(target.data.view_as(y_pred_g)).long().cpu().sum().item()
                total_test_loss += user_test_loss
                total_correct_g += user_correct_g
                total_samples += user_total_samples

        # 计算全局的平均损失和准确率
        # 防止样本数为0的情况
        if total_samples > 0:
            average_test_loss = total_test_loss / total_samples
            accuracy_g = 100.00 * total_correct_g / total_samples
        else:
            average_test_loss = 0
            accuracy_g = 0

    return accuracy_g, average_test_loss
def test_img_mix(net_l, net_g, gate, datatest, args, test_loader):
    with torch.no_grad():
        net_g.eval()
        net_l.eval()
        gate.eval()
        total_test_loss = 0
        total_correct_g = 0
        total_samples = 0

        # 假设 test_loader 可能是 dict 或者 dataloader
        if isinstance(test_loader, dict):
            for user_idx, data_loader in test_loader.items():
                user_total_samples = 0  # Initialize inside loop for users
                user_test_loss = 0
                user_correct_g = 0

                for data, target in data_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    user_total_samples += data.size(0)  # Count samples correctly
                    gate_weight = gate(data.float())
                    log_probs = gate_weight * net_l(data.float()) + (1 - gate_weight) * net_g(data.float())
                    loss = nn.CrossEntropyLoss()(log_probs, target)
                    user_test_loss += loss.item()

                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    user_correct_g += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

                total_test_loss += user_test_loss
                total_correct_g += user_correct_g
                total_samples += user_total_samples  # Add user samples to total
        elif isinstance(test_loader, list):
            for loader in test_loader:
                user_total_samples = 0  # Initialize inside loop for loaders
                user_test_loss = 0
                user_correct_g = 0

                for data, target in loader:
                    data, target = data.to(args.device), target.to(args.device)
                    user_total_samples += data.size(0)  # Count samples correctly
                    gate_weight = gate(data.float())
                    log_probs = gate_weight * net_l(data.float()) + (1 - gate_weight) * net_g(data.float())
                    loss = nn.CrossEntropyLoss()(log_probs, target)
                    user_test_loss += loss.item()

                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    user_correct_g += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

                total_test_loss += user_test_loss
                total_correct_g += user_correct_g
                total_samples += user_total_samples  # Add loader samples to total
        else:
            # 如果 test_loader 不是字典，则假设它是一个 dataloader
            user_total_samples = 0  # Initialize inside loop for dataloader
            user_test_loss = 0
            user_correct_g = 0

            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                user_total_samples += data.size(0)  # Count samples correctly
                gate_weight = gate(data.float())
                log_probs = gate_weight * net_l(data.float()) + (1 - gate_weight) * net_g(data.float())
                loss = nn.CrossEntropyLoss()(log_probs, target)
                user_test_loss += loss.item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                user_correct_g += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

            total_test_loss += user_test_loss
            total_correct_g += user_correct_g
            total_samples += user_total_samples  # Add dataloader samples to total

        # 防止样本数为0的情况
        if total_samples > 0:
            average_test_loss = total_test_loss / total_samples
            accuracy = 100.00 * total_correct_g / total_samples
        else:
            average_test_loss = 0
            accuracy = 0

    return accuracy, average_test_loss


# def test_img(net_g, datatest, args,test_dataloader):
#     with torch.no_grad():
#         net_g.eval()
#         #for i in range(len(net_l)):
#         #    net_l[i].eval()
#         # testing
#         test_loss = 0
#         test_loss_local = 0
#         correct_g = 0
#         correct_local = 0
#         total_test_loss = 0
#         total_correct_g = 0
#         total_samples = 0
#
#         # datasize = len(datatest)
#         # sub_idxs = np.random.choice(datasize,int(0.1*datasize),replace=False)
#         # datatest = DatasetSplit(datatest,sub_idxs)
#         # data_loader = DataLoader(datatest, batch_size=1)
#         # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#         # l = len(data_loader)
#
#         for user_id, loader in test_dataloader.items():
#             user_test_loss = 0
#             user_correct_g = 0
#             user_total_samples = len(loader.dataset)
#
#             for data, target in loader:
#                 data, target = data.to(args.device), target.to(args.device)
#                 log_probs_g = net_g(data.float())
#                 user_test_loss += nn.CrossEntropyLoss(log_probs_g, target).item()
#
#                 y_pred_g = log_probs_g.data.max(1, keepdim=True)[1]
#                 correct_g += y_pred_g.eq(target.data.view_as(y_pred_g)).long().cpu().sum() #Computes element-wise equality
#
#             # 汇总每个用户的数据
#             total_test_loss += user_test_loss
#             total_correct_g += user_correct_g
#             total_samples += user_total_samples
#         # 计算全局的平均损失和准确率
#         average_test_loss = total_test_loss / total_samples
#         accuracy_g = 100.00 * total_correct_g / total_samples
#     return accuracy_g.item(), average_test_loss#test_loss
#
#
# def test_img_mix(net_l, net_g, gate, datatest, args, test_loader):
#     with torch.no_grad():
#         net_g.eval()
#         net_l.eval()
#         gate.eval()
#         test_loss = 0
#         correct = 0
#         correct_local = 0
#         total_test_loss = 0
#         total_correct_g = 0
#         total_samples = 0
#
#         for user_idx, data_loader in test_loader.items():
#             user_total_samples = len(data_loader.dataset)
#             user_test_loss = 0
#             user_correct_g = 0
#             for data, target in data_loader:
#                 # if args.gpu != -1:
#                 #     data, target = data.to(args.device), target.to(args.device)
#                 gate_weight = gate(data)
#                 log_probs = gate_weight * net_l(data) + (1-gate_weight) * net_g(data)
#                 user_test_loss += nn.CrossEntropyLoss(log_probs, target).item()
#                 y_pred = log_probs.data.max(1,keepdim=True)[1]
#                 user_correct_g += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()
#
#             total_test_loss += user_test_loss
#             total_correct_g += user_correct_g
#             total_samples += user_total_samples
#
#         average_test_loss = total_test_loss / total_samples
#         accuracy = 100.00 * total_correct_g / total_samples
#     return accuracy.item(), average_test_loss
#
