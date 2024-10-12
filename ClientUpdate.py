import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np
import time
import random
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# from shakespeare_data_model import TextDataset
from Models import CNNFashion, CNNCifar, GateModel, Net_adult,Net_purchase
from data_preprocess import data_set


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def generate_client_scores(moe_gate_model, client_data_cifar, client_data_fashion):
    moe_gate_model.eval()
    with torch.no_grad():
        client_scores = moe_gate_model(client_data_cifar, client_data_fashion)
    return client_scores


class ClientUpdate(object):
    def __init__(self, args, model_type, writer, train_set=None, test_set=None, idxs_train=None, idxs_val=None,
                 idxs_test=None, user_train_dataloaders=None, user_val_dataloaders=None, user_test_dataloaders=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss() #nn.NLLLoss()
        # self.train_set = DatasetSplit(train_set, idxs_train)
        # self.val_set = DatasetSplit(train_set, idxs_val)
        # self.test_set = DatasetSplit(test_set, idxs_test)
        if args.seperate_mix:
            self.train_set = train_set
            self.val_set = train_set
            self.test_set = test_set
            self.ldr_train = user_train_dataloaders
            self.ldr_val = user_val_dataloaders
            self.ldr_test = user_test_dataloaders
        else:
            self.train_set = DatasetSplit(train_set, idxs_train)
            self.val_set = DatasetSplit(train_set, idxs_val)
            self.test_set = DatasetSplit(test_set, idxs_test)
            self.ldr_train = DataLoader(self.train_set, batch_size=64, shuffle=True)
            self.ldr_val = DataLoader(self.val_set, batch_size=32, shuffle=True)
            self.ldr_test = DataLoader(self.test_set, batch_size=1, shuffle=True)

        if model_type == 'cnn_cifar':
            self.model = CNNCifar(args).to(args.device)
        elif model_type == 'resnet_cifar':
            self.model = models.resnet18(pretrained=False, num_classes=10).to(args.device)
        elif model_type == 'cnn_mnist':
            self.model = CNNFashion(args).to(args.device)
        elif model_type == 'purchase':
            self.model = Net_purchase().to(args.device)
        elif model_type == 'adult':
            self.model = Net_adult().to(args.device)
        # dataset_length = len(self.train_set)


        self.writer = writer #SummaryWriter()  # Initialize TensorBoard writer
        self.loaders_train = user_train_dataloaders
        self.loaders_val = user_val_dataloaders


    def train(self, net, n_epochs, learning_rate):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        epoch_loss = []
        epoch_train_accuracy=[]
        correct = 0

        for epoch in range(n_epochs):
            net.train()
            batch_loss = []
            correct = 0  # 在每个 epoch 开始前重置正确计数
            total_samples = 0  # 用于累积样本总数

            if isinstance(self.ldr_train, list):
                loader = self.ldr_train[0]
                for images, labels in loader:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images.float())
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    batch_loss.append(loss.item())
                    _, predicted = torch.max(log_probs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)  # 累积样本总数
            elif isinstance(self.ldr_train, dict):
                for user_id, loader in self.ldr_train.items():
                    for images, labels in loader:
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(images.float())
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        optimizer.step()
                        batch_loss.append(loss.item())
                        _, predicted = torch.max(log_probs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total_samples += labels.size(0)  # 累积样本总数
            else:
                for images, labels in self.ldr_train:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images.float())
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    batch_loss.append(loss.item())
                    _, predicted = torch.max(log_probs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)  # 累积样本总数


            # 正确计算训练准确率
            train_accuracy = 100.0 * correct / total_samples
            epoch_train_accuracy.append(train_accuracy)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), epoch_loss[-1], epoch_train_accuracy[-1]

    def train_finetune(self, net, n_epochs, learning_rate, val):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        patience = 10
        epoch_loss = []
        epoch_train_accuracy = []
        model_best = net.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        total_time=0
        total_communication_rounds=0
        start_time = time.time()
        for epoch in range(n_epochs):
            total_communication_rounds += 1
            net.train()
            batch_loss = []
            correct = 0
            ldr_train = self.ldr_train
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images.float())
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs.data, 1)
                correct += (predicted == labels).sum().item()
            train_accuracy = 100.00 * correct / len(ldr_train.dataset)
            epoch_train_accuracy.append(train_accuracy)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch % 5 == 0:
                val_acc, val_loss = self.validate(net, val)
                net.train()
                # print(f"Epoch {epoch} | Train Loss: {epoch_loss[-1]} | Val Loss: {val_loss} | Val Acc: {val_acc}")
                if val_loss < val_loss_best - 0.01:
                    counter = 0
                    model_best = net.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    train_acc_best = train_accuracy
                    # print("Update Train Finetune Iter: %d | Validation best acc: %.2f" % (epoch, val_acc_best))
                else:
                    counter += 1
                if counter == patience:
                    return model_best, epoch_loss[-1], val_acc_best, train_acc_best
            # 调整学习率
            if epoch > 0 and epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
            end_time = time.time()
            round_time = end_time - start_time
            total_time += round_time
            # print(f"Communication round {total_communication_rounds} took {round_time:.2f} seconds")
            # Log the values to TensorBoard
            if isinstance(net,CNNCifar):
                self.writer.add_scalar('cnn_cifar_train_finetuneLoss/train_finetune', epoch_loss[-1], total_communication_rounds)
                self.writer.add_scalar('cnn_cifar_train_finetuneLoss/validation_finetune', val_loss_best, total_communication_rounds)
                self.writer.add_scalar('cnn_cifar_train_finetuneAccuracy/validation_finetune', val_acc_best, total_communication_rounds)
                self.writer.add_scalar('cnn_cifar_train_finetuneAccuracy/train_finetune', train_acc_best, total_communication_rounds)
            elif isinstance(net, CNNFashion):
                self.writer.add_scalar('cnn_mnist_train_finetuneLoss/train_finetune', epoch_loss[-1],
                                       total_communication_rounds)
                self.writer.add_scalar('cnn_mnist_train_finetuneLoss/validation_finetune', val_loss_best,
                                       total_communication_rounds)
                self.writer.add_scalar('cnn_mnist_train_finetuneAccuracy/validation_finetune', val_acc_best,
                                       total_communication_rounds)
                self.writer.add_scalar('cnn_mnist_train_finetuneAccuracy/train_finetune', train_acc_best,
                                       total_communication_rounds)

        # 结束后，输出总时间和通信轮数
        # print(f"Total communication rounds: {total_communication_rounds}")
        # print(f"Total time taken for training: {total_time:.2f} seconds")

        return model_best, epoch_loss[-1], val_acc_best, train_acc_best

    def train_mix(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop, learning_rate, val, idxs_users, dict_users, FL_params):
        net_local.train()
        net_global.train()
        gate.train()

        if train_gate_only:
            optimizer = torch.optim.Adam(gate.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(list(net_local.parameters()) + list(gate.parameters()), lr=learning_rate)

        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        local_best = net_local.state_dict()
        global_best = net_global.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0


        for epoch in range(self.args.local_epoch):
            # print('Mix Step Gating Epoch {}/{}'.format(epoch + 1, n_epochs))
            net_local.train()
            net_global.train()
            gate.train()

            batch_loss = []
            if isinstance(self.ldr_train, list):
                loader = self.ldr_train[0]
                for images, labels in loader:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net_local.zero_grad()
                    net_global.zero_grad()
                    gate.zero_grad()

                    if isinstance(gate, GateModel):
                        gate_weight = gate(images.float(), len(self.loaders_train), loader, dict_users,
                                           net_local, FL_params)
                    else:
                        gate_weight = gate(images.float())
                    local_probs = net_local(images.float())
                    global_probs = net_global(images.float())

                    if gate_weight.dim() == 1:
                        gate_weight = gate_weight.expand(-1, local_probs.size(1))

                    log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
            elif isinstance(self.ldr_train, dict):
                for user_id, loader in self.ldr_train.items():
                    for images, labels in loader:
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net_local.zero_grad()
                        net_global.zero_grad()
                        gate.zero_grad()

                        if isinstance(gate, GateModel):
                            gate_weight = gate(images.float(), len(self.loaders_train), loader, dict_users,
                                               net_local, FL_params)
                        else:
                            gate_weight = gate(images.float())
                        local_probs = net_local(images.float())
                        global_probs = net_global(images.float())

                        if gate_weight.dim() == 1:
                            gate_weight = gate_weight.expand(-1, local_probs.size(1))

                        log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        early_stop=True
        if early_stop and epoch % 2 == 0:
            for idx in idxs_users:
                val_acc, val_loss = self.validate_mix(net_local, net_global, gate, val, idxs_users, dict_users, FL_params)
                net_local.train()
                net_global.train()
                gate.train()

                # print(f"Train Loss: {epoch_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}")
                if val_loss < val_loss_best - 0.01:
                    counter = 0
                    gate_best = gate.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    local_best = net_local.state_dict()
                    global_best = net_global.state_dict()
                    # print("Train mix Iter %d | Validation best acc: %.2f" % (epoch, val_acc_best))
                else:
                    counter += 1
                if counter == patience:
                    return gate_best, local_best, global_best, epoch_loss[-1], val_acc_best
                # epoch = epoch+1
            if isinstance(net_local, CNNCifar):
                self.writer.add_scalar('cnn_cifar_gating_mix_loss/epochloss', epoch_loss[-1],
                                       epoch)
                self.writer.add_scalar('cnn_cifar_gating_loss/validation', val_loss_best,
                                       epoch)
                self.writer.add_scalar('cnn_cifar_gating_mix_acc/validation', val_acc_best,
                                       epoch)

            elif isinstance(net_local, CNNFashion):
                self.writer.add_scalar('cnn_mnist_mix_Loss/train_finetune', epoch_loss[-1],
                                       epoch)
                self.writer.add_scalar('cnn_mnist_mix_Loss/validation_finetune', val_loss_best,
                                       epoch)
                self.writer.add_scalar('cnn_mnist_mix_Accuracy/validation_finetune', val_acc_best,
                                       epoch)

        return gate_best, local_best, global_best, epoch_loss[-1], val_acc_best

    def validate(self, net, val):
        # 根据val的值选择验证集或测试集的dataloader
        dataloader = self.ldr_val if val else self.ldr_test

        with torch.no_grad():
            net.eval()
            total_val_loss = 0
            total_correct = 0
            total_samples = 0

            # 检查dataloader的类型
            if isinstance(dataloader, dict):
                # 如果是字典，遍历每个用户的数据加载器
                for user_id, loader in dataloader.items():
                    for data, target in loader:
                        data, target = data.to(self.args.device), target.to(self.args.device)
                        log_probs = net(data.float())
                        batch_loss = self.loss_func(log_probs, target).item()
                        total_val_loss += max(batch_loss, 1e-10)
                        y_pred = log_probs.data.max(1, keepdim=True)[1]
                        total_correct += y_pred.eq(target.data.view_as(y_pred)).long().sum().item()
                        total_samples += target.size(0)

            elif isinstance(dataloader, list):
                loader = dataloader[0]
                for data, target in loader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    log_probs = net(data.float())
                    batch_loss = self.loss_func(log_probs, target).item()
                    total_val_loss += max(batch_loss, 1e-10)
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    total_correct += y_pred.eq(target.data.view_as(y_pred)).long().sum().item()
                    total_samples += target.size(0)

            else:
                # 如果是DataLoader，直接使用
                for data, target in dataloader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    log_probs = net(data.float())
                    batch_loss = self.loss_func(log_probs, target).item()
                    total_val_loss += max(batch_loss, 1e-10)
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    total_correct += y_pred.eq(target.data.view_as(y_pred)).long().sum().item()
                    total_samples += target.size(0)

            # 计算平均损失和准确率
            average_val_loss = total_val_loss / total_samples
            accuracy = 100.00 * total_correct / total_samples

            # # Log the values to TensorBoard
            # self.writer.add_scalar('validate_Loss/validation', average_val_loss, self.epoch)
            # self.writer.add_scalar('validate_Accuracy/validation', accuracy, self.epoch)

        return accuracy, average_val_loss

    def validate_mix(self, net_l, net_g, gate, val, idxs_users, dict_users, FL_params):
        dataloader = self.ldr_val if val else self.ldr_test
        with torch.no_grad():
            net_l.eval()
            net_g.eval()
            gate.eval()
            val_loss = 0
            correct = 0
            total_samples = 0

            if isinstance(dataloader, list):
                for loader in dataloader:
                    for data, target in loader:
                        data, target = data.to(self.args.device), target.to(self.args.device)
                        gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
                                           net_l, FL_params) if isinstance(gate, GateModel) else gate(data.float())
                        local_probs = net_l(data.float())
                        global_probs = net_g(data.float())
                        log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
                        val_loss += self.loss_func(log_probs, target).item()
                        y_pred = log_probs.data.max(1, keepdim=True)[1]
                        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()
                        total_samples += target.size(0)

            elif isinstance(dataloader, dict):
                for user_id, loader in dataloader.items():
                    for data, target in loader:
                        data, target = data.to(self.args.device), target.to(self.args.device)
                        gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
                                           net_l, FL_params) if isinstance(gate, GateModel) else gate(data.float())
                        local_probs = net_l(data.float())
                        global_probs = net_g(data.float())
                        log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
                        val_loss += self.loss_func(log_probs, target).item()
                        y_pred = log_probs.data.max(1, keepdim=True)[1]
                        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()
                        total_samples += target.size(0)

            else:
                loader = dataloader
                for data, target in loader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
                                       net_l, FL_params) if isinstance(gate, GateModel) else gate(data.float())
                    local_probs = net_l(data.float())
                    global_probs = net_g(data.float())
                    log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
                    val_loss += self.loss_func(log_probs, target).item()
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()
                    total_samples += target.size(0)

            # 计算平均验证损失和准确率
            val_loss /= total_samples
            accuracy = 100.00 * correct / total_samples

            return accuracy, val_loss

    # def validate_mix(self,net_l, net_g, gate, val, idxs_users, dict_users, FL_params):
    #     dataloader = self.ldr_val if val else self.ldr_test
    #     with torch.no_grad():
    #         net_l.eval()
    #         net_g.eval()
    #         gate.eval()
    #         val_loss = 0
    #         correct = 0
    #         epoch = 0
    #         if isinstance(dataloader, list):
    #             loader = dataloader[0]
    #             for data, target in loader:
    #                 data, target = data.to(self.args.device), target.to(self.args.device)
    #                 if isinstance(gate, GateModel):
    #                     gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
    #                                        net_l, FL_params)
    #                 else:
    #                     gate_weight = gate(data.float())
    #                 # gate_weight = gate(data.float())
    #                 # ,features, classes = acculumate_feature(finetuned[i], user_train_dataloaders[i], stop=10)
    #                 # gate_weight = gate(data.float(), features, classes, idxs_users, self.loaders_train, dict_users, net_l, FL_params)
    #                 local_probs = net_l(data.float())
    #                 global_probs = net_g(data.float())
    #                 log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
    #                 val_loss += self.loss_func(log_probs, target).item()
    #                 y_pred = log_probs.data.max(1, keepdim=True)[1]
    #                 correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    #                 epoch = epoch
    #         elif isinstance(dataloader, dict):
    #             for user_id, loader in dataloader.items():
    #                 for data, target in loader:
    #                     data, target = data.to(self.args.device), target.to(self.args.device)
    #                     if isinstance(gate, GateModel):
    #                         gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
    #                                            net_l, FL_params)
    #                     else:
    #                         gate_weight = gate(data.float())
    #                     local_probs = net_l(data.float())
    #                     global_probs = net_g(data.float())
    #                     log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
    #                     val_loss += self.loss_func(log_probs, target).item()
    #                     y_pred = log_probs.data.max(1, keepdim=True)[1]
    #                     correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    #                     epoch = epoch + 1
    #         else:
    #             loader = dataloader
    #             for data, target in loader:
    #                 data, target = data.to(self.args.device), target.to(self.args.device)
    #                 if isinstance(gate, GateModel):
    #                     gate_weight = gate(data.float(), len(self.loaders_train), loader, dict_users,
    #                                        net_l, FL_params)
    #                 else:
    #                     gate_weight = gate(data.float())
    #                 local_probs = net_l(data.float())
    #                 global_probs = net_g(data.float())
    #                 log_probs = gate_weight * local_probs + (1 - gate_weight) * global_probs
    #                 val_loss += self.loss_func(log_probs, target).item()
    #                 y_pred = log_probs.data.max(1, keepdim=True)[1]
    #                 correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    #                 epoch = epoch + 1
    #
    #         val_loss /= len(loader.dataset)
    #         accuracy = 100.00 * correct / len(loader.dataset)
    #         # val_loss /= len(dataloader)
    #         # accuracy = 100.00 * correct / len(dataloader)
    #         # print(f'validate mix accuracy, val_loss', accuracy.item(), val_loss)
    #
    #         # Log the values to TensorBoard
    #         self.writer.add_scalar('validate_mix_Loss/validation_mix', val_loss, epoch)
    #         self.writer.add_scalar('validate_mix_Accuracy/validation_mix', accuracy.item(), epoch)
    #         # print('accuracy, val_loss', accuracy.item(), val_loss)
    #
    #         return accuracy.item(), val_loss