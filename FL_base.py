from pathlib import Path

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import time
import pandas as pd
# ourself libs
from model_initiation import model_init
from data_preprocess import data_set


class base_warmup():
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        ''' base class for warmup scheduler
        Args:
            optimizer: adjusted optimizer
            warm_step: total number of warm_step,(batch num)
            warm_lr: start learning rate of warmup
            dst_lr: init learning rate of train stage eg. 0.01
        '''
        assert warm_lr < dst_lr, "warmup lr must smaller than init lr"
        self.optimizer = optimizer
        self.warm_lr = warm_lr
        self.init_lr = dst_lr
        self.warm_step = warm_step
        self.stepped = 0
        if self.warm_step:
            self.optimizer.param_groups[0]['lr'] = self.warm_lr

    def step(self):
        self.stepped += 1

    def if_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False

    def set2initial(self):
        ''' Reset the learning rate to initial lr of training stage '''
        self.optimizer.param_groups[0]['lr'] = self.init_lr

    @property
    def now_lr(self):
        return self.optimizer.param_groups[0]['lr']

class linear_warmup_scheduler(base_warmup):
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        super().__init__(optimizer, warm_step, warm_lr, dst_lr)
        if (self.warm_step <= 0):
            self.inc = 0
        else:
            self.inc = (self.init_lr - self.warm_lr) / self.warm_step

    def step(self) -> bool:
        if (not self.stepped < self.warm_step): return False
        self.optimizer.param_groups[0]['lr'] += self.inc
        super().step()
        return True

    def still_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False

def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    """
    init_global_model：初始化的模型，model_init(FL_params.data_name)下载得到
    client_data_loaders：被抽取的客户端数据加载器（训练集）
    test_loader：测试集
    FL_params：FL对象
    """

    global_model = init_global_model
    project_dir = Path(__file__).resolve().parent
    save_info = project_dir / 'ckpt' / FL_params.model_name

    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
        # 聚合，更新全局模型的参数，会应用到下一轮训练当中
        global_model = fedavg(client_models)
        print("Global Federated Learning epoch = {}".format(epoch))

        net, val_acc = test(global_model, test_loader)
        save_flag = save_net(net, val_acc, FL_params.save_acc, save_info, epoch)
        if save_flag:
            break

    return net, val_acc, epoch


def FL_Finetuned(init_global_model, client_data_loaders, test_loader, FL_params):
    global_model = init_global_model
    project_dir = Path(__file__).resolve().parent
    save_info = project_dir / 'ckpt' / 'finetuned' / FL_params.model_name

    epoch_acc = []

    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
        # 聚合，更新全局模型的参数，会应用到下一轮训练当中
        global_model = fedavg(client_models)
        print("Global Federated Learning epoch = {}".format(epoch))

        net, val_acc = test(global_model, test_loader)

        epoch_acc.append((epoch, val_acc))
        save_flag = save_net(net, val_acc, FL_params.save_re_acc, save_info, epoch)
        if save_flag:
            break

    epochs, accuracies = zip(*epoch_acc)

    # 将数据存储到Excel表中
    data = {'Epoch': epochs, 'Accuracy': accuracies}
    df = pd.DataFrame(data)
    excel_file = FL_params.data_name + '_' + FL_params.model_name + '_finetuned.xlsx'
    df.to_excel('./excel/' + excel_file, index=False)

    return net, val_acc, epoch

def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    global_model = init_global_model
    project_dir = Path(__file__).resolve().parent
    save_info = project_dir / 'ckpt' / 'retrained' / FL_params.model_name

    epoch_acc = []

    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
        # 聚合，更新全局模型的参数，会应用到下一轮训练当中
        global_model = fedavg(client_models)
        print("Global Federated Learning epoch = {}".format(epoch))

        net, val_acc = test(global_model, test_loader)

        epoch_acc.append((epoch, val_acc))
        save_flag = save_net(net, val_acc, FL_params.save_re_acc, save_info, epoch)
        if save_flag:
            break
    epochs, accuracies = zip(*epoch_acc)

    # 将数据存储到Excel表中
    data = {'Epoch': epochs, 'Accuracy': accuracies}
    df = pd.DataFrame(data)
    excel_file = FL_params.data_name + '_' + FL_params.model_name + '_retrained.xlsx'
    df.to_excel('./excel/' + excel_file, index=False)

    return net, val_acc, epoch

"""
Function：
For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
NOTE:The global model inputed is the global model for the previous round
    The output client_Models is the model that each user trained separately.
"""


# training sub function
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):
    # 使用每个client的模型、优化器、数据，以client_models为训练初始模型，使用client用户本地的数据和优化器，更新得到update——client_models
    # Note：需要注意的一点是，global_train_once只是在全局上对模型的参数进行一次更新
    # update_client_models = list()
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")

    client_models = []
    client_sgds = []

    # 为每一个客户端创建一个SGD优化器
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9, weight_decay=5e-4, nesterov=True))

    for client_idx in range(FL_params.N_client):

        model = client_models[client_idx]  # 模型
        optimizer = client_sgds[client_idx]  # 优化器

        # 创建了一个学习率调度器，使用余弦退火调整学习率，在训练过程中，学习率将随着时间的推移而逐渐降低。
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, FL_params.local_epoch, eta_min=0)

        # 创建了一个学习率预热调度器，用于在训练开始阶段逐渐增加学习率，以帮助模型更快地收敛
        warmup_scheduler = linear_warmup_scheduler(optimizer, 0, 10e-5, FL_params.local_lr)

        model.to(device)
        model.train()

        # local training：客户端在本地对私有数据进行训练
        for local_epoch in range(FL_params.local_epoch):

            train_loss = 0
            correct = 0
            total = 0

            """
            enumerate(client_data_loaders[client_idx]): 
            这个函数返回一个迭代器，产生一对索引和元素，其中索引是迭代次数（批次的索引），元素是从数据加载器中获取的批量数据。
            在这里，batch_idx 是批次的索引，而 (data, target) 是从数据加载器中获取的一批输入数据和对应的目标标签。
            """
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()

                data = data.to(device)
                target = target.to(device)

                # 清零之前的梯度，因为默认情况下 PyTorch 会累积梯度。
                optimizer.zero_grad()

                # pred = model(data): 使用模型进行前向传播，得到预测值。
                pred = model(data)

                # 定义交叉熵损失函数，用于计算模型的预测值与真实标签之间的差异。
                criteria = nn.CrossEntropyLoss()

                # 计算损失
                loss = criteria(pred, target)

                # 反向传播，计算梯度
                loss.backward()

                # 根据梯度更新模型参数
                optimizer.step()

                if if_warmup:
                    warmup_scheduler.step()

                # 累加当前批次的损失值
                train_loss += loss.item()

                # 根据模型输出，获取预测的类别
                _, predicted = pred.max(1)

                # 增加累计的样本数
                total += pred.size(0)
                # 增加预测正确的样本数
                correct += predicted.eq(target).sum().item()

            if local_epoch == FL_params.local_epoch - 1 and client_idx == FL_params.N_client - 1:
                # 计算当前训练周期的平均损失。
                training_loss = train_loss / (batch_idx + 1)
                # 计算当前训练周期的准确率。
                training_acc = correct / total
                print("Train:")
                print("Train Loss=%.8f, Train acc=%.8f" % (training_loss, training_acc))
            if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
                scheduler.step()

            # if val_acc * 100 > save_acc:
            #     _save_acc = val_acc * 100
            #     save_path = save_info / ('seed' + str(seed) + '_acc' + str(_save_acc)[0:5] +
            #                              '_epoch' + str(epoch) +
            #                              time.strftime("_%Y-%m-%d %H-%M-%S", time.localtime()) + '.pth')
            #     print('save path', save_path)
            #     torch.save(net.state_dict(), save_path)

        # if (FL_params.train_with_test and client_idx == FL_params.N_client - 1):
        # print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
        # test(model, test_loader)

        # if(FL_params.use_gpu*FL_params.cuda_state):
        # 更新模型（后续聚合）
        model.to(device_cpu)
        client_models[client_idx] = model

    return client_models


"""
Function：
Test the performance of the model on the test set
"""


def test(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    loss = test_loss / num_val_steps
    return net, val_acc


def save_net(net, val_acc, save_acc, save_info, epoch):
    save_flag = False
    if val_acc * 100 > save_acc:
        _save_acc = val_acc * 100
        save_path = save_info / ('seed' + '_acc' + str(_save_acc)[0:5] +
                                 '_epoch' + str(epoch) +
                                 time.strftime("_%Y-%m-%d %H-%M-%S", time.localtime()) + '.pth')
        print('save path', save_path)
        torch.save(net.state_dict(), save_path)
        save_flag = True
    return save_flag


"""
Function：
FedAvg：聚合
"""


def fedavg(local_models):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
    local_model_weights : tensor or array
        DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)

    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """
    # N = len(local_models)
    # new_global_model = copy.deepcopy(local_models[0])
    # print(len(local_models))

    # 将第一个模型深拷贝
    global_model = copy.deepcopy(local_models[0])

    # 用于存储参数的平均值
    avg_state_dict = global_model.state_dict()

    # 用于存储本地模型的状态
    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    # 遍历每一层，每到一层都先将第一个数据初始化为0（*0）
    # 针对所有本地模型，计算每一层的参数平均值
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] = avg_state_dict[layer] / len(local_models)

    # 返回更新后的全局模型
    global_model.load_state_dict(avg_state_dict)
    return global_model
