# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:15:13 2020

@author: jojo
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

# ourself libs
from fedbabygpt import load_data, split_data, train_model, generate_text, BabyGPTmodel, GPTConfig
from data_preprocess import Net_cifar10_new, Net_mnist_new

# from FL_base import test
from FederatedAveraging import FedAvg
from ClientUpdate import ClientUpdate
from torch.utils.tensorboard import SummaryWriter

def mix_Train(opt_in,dataset_train,dataset_test,dict_users,dict_users_val,dict_users_test,net_glob_fedAvg,writer,FL_params):
    # training
    val_loss_best = np.inf #表示+∞
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
        # TODO frac=5
        # m = max(int(FL_params.frac), 1)
        # idxs_users = np.random.choice(opt_in, m, replace=False)  # choose opt-in clients
        idxs_users = opt_in
        for idx in idxs_users:
            print("FedAvg client %d" % (idx))

            client = ClientUpdate(args=FL_params, train_set=dataset_train, test_set=dataset_test,
                                  idxs_train=dict_users[idx], idxs_val=dict_users_val[idx],
                                  idxs_test=dict_users_test[idx])

            # train FedAvg
            w_glob_fedAvg, train_loss_idx = client.train(net=copy.deepcopy(net_glob_fedAvg).to(FL_params.device),
                                                         n_epochs=FL_params.local_epoch, learning_rate=5e-5)

            w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))
            train_loss.append(train_loss_idx)
            # Weigh models by client dataset size
            alpha.append(len(dict_users[idx]))

            if (n_iter % 4 == 0):
                val_acc_fed, val_loss_fed = client.validate(net=net_glob_fedAvg, val=True)
                val_acc.append(val_acc_fed)
                val_loss.append(val_loss_fed)

        # update global model weights
        train_loss_avg = sum(train_loss) / len(train_loss)
        writer.add_scalar('fedAvg_train_loss', train_loss_avg, n_iter)

        if (n_iter % 4 == 0):
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
    return net_glob_fedAvg, writer

def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    # if(FL_params.if_retrain == True):
    #     raise ValueError('FL_params.if_retrain should be set to False, if you want to train, not retrain FL model')
    if (FL_params.if_unlearning == True):
        raise ValueError(
            'FL_params.if_unlearning should be set to False, if you want to train, not unlearning FL model')

    all_global_models = list()
    all_client_models = list()
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if (FL_params.mix_experts):
        client_models = copy.deepcopy(init_global_model)
        client_cifar10_model = []
        client_mnist_model = []
        for client_model in client_models:
            if type(client_model) is Net_cifar10_new:
                client_cifar10_model.append(client_model)
                # print('111',client_cifar10_model)
            elif type(client_model) is Net_mnist_new:
                client_mnist_model.append(client_model)
                # print('2')
        # print('1', client_cifar10_model)
        for epoch in range(FL_params.global_epoch):
            cifar10_client_models = global_train_once(client_cifar10_model[0], client_data_loaders, test_loader, FL_params)
            mnist_client_models =global_train_once(client_mnist_model[0], client_data_loaders, test_loader, FL_params)
            # print(cifar10_client_models)
            # for cm in client_models:
            #     for i in range(10):
            #         client_models = global_train_once(cm, client_data_loaders[i], test_loader, FL_params)

            cifar10_global_model = fedavg(cifar10_client_models)
            mnist_global_model = fedavg(mnist_client_models)
            print("Global Federated Learning epoch = {}".format(epoch))

            all_global_models.append(copy.deepcopy(cifar10_global_model))
            all_global_models.append(copy.deepcopy(mnist_global_model))
    else:
        global_model = init_global_model

        all_global_models.append(copy.deepcopy(global_model))

        for epoch in range(FL_params.global_epoch):
            client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
            # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
            # IMPORTANT: It is IMPORTANT to note here that global_train_once is trained directly on the input client_models during training, so the output's client_models are the same set of models as the input's client_models, except that the input is untrained while the output is trained.
            # Therefore, in order to implement Federated Unlearning, we need to save the models in Client -- Models before global Train.You can use DeepCopy, or hard disk IO.
            all_client_models += client_models
            global_model = fedavg(client_models)
            # print(30*'^')
            print("Global Federated Learning epoch = {}".format(epoch))
            # test(global_model, test_loader)
            # print(30*'v')
            # print(len(all_client_models))
            all_global_models.append(copy.deepcopy(global_model))

    return all_global_models, all_client_models


def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    # device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
    if (FL_params.if_retrain == False):
        raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')
    print('FL_params.forget_client_idx', FL_params.forget_client_idx)
    # if(FL_params.forget_client_idx[idx_client] not in range(FL_params.N_client)for idx_client in FL_params.forget_client_idx):
    #     raise ValueError('FL_params.forget_client_idx should be in [{}], if you want to use standard FL train with forget the certain client dataset.'.format(range(FL_params.N_client)))
    # forget_idx= FL_params.forget_idx
    print('\n')
    print(5 * "#" + "  Federated Retraining Start  " + 5 * "#")
    # std_time = time.time()
    print("Federated Retrain with Forget Client NO.{}".format(FL_params.forget_client_idx))
    retrain_GMs = list()
    all_client_models = list()
    retrain_GMs.append(copy.deepcopy(init_global_model))
    global_model = init_global_model.to(FL_params.device)
    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。：It is important to note that global_train_once is trained directly on the input client_models during training, so the output's client_models are the same set of models as the input's client_models, except that the input is untrained while the output is trained.
        #   IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
        # IMPORTANT: Therefore, in order to implement Federated Unlearning, we need to save the models in Client -- Models before global Train.You can use DeepCopy, or hard disk IO.
        global_model = fedavg(client_models).to(FL_params.device)
        # print(30*'^')
        print("Global Retraining epoch = {}".format(epoch))
        # test(global_model, test_loader)
        # print(30*'v')
        retrain_GMs.append(copy.deepcopy(global_model))
        # print("retrain_GMs len",len(retrain_GMs))

        all_client_models += client_models
    # end_time = time.time()
    print(5 * "#" + "  Federated Retraining End  " + 5 * "#")
    return retrain_GMs


def target_to_output_shape(target, output_shape):
    batch_size, num_channels, height, width = output_shape
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_channels).float()
    target_one_hot = target_one_hot.view(batch_size, num_channels, 1, 1)
    target_one_hot = target_one_hot.expand(batch_size, num_channels, height, width)
    return target_one_hot


"""
Function：
For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
NOTE:The global model inputed is the global model for the previous round
    The output client_Models is the model that each user trained separately.
"""


def global_train_once(global_model, client_data_loader, test_loader, FL_params):
    # 使用每个client的模型、优化器、数据，以client_models为训练初始模型，使用client用户本地的数据和优化器，更新得到upodate——client_models
    # Note：需要注意的一点是，global_train_once只是在全局上对模型的参数进行一次更新
    # Using the model, optimizer, and data of each client, training the initial model with client_models, updating the UPODate -- client_models using the client user's local data and optimizer
    # Note: It is important to Note that global_train_once is only a global update to the parameters of the model
    # update_client_models = list()
    # device = torch.device("cuda:2" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    # if (FL_params.mix_experts):
    #     client_models = []
    #     for m in global_model:
    #         m.to(device)
    #         # client_models = []
    #         # client_sgds = []
    #         # for ii in range(FL_params.N_client):
    #         #     client_models.append(copy.deepcopy(m))
    #         #     client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))
    #         #
    #         # # for client_idx in range(FL_params.N_client):
    #         # #     model = client_models[client_idx]  # .to(device)
    #         #
    #         # if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or (
    #         #         (FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
    #         #     continue
    #
    #         # optimizer = client_sgds[client_idx]
    #
    #         # model
    #         m.train()
    #
    #         # local training
    #         for local_epoch in range(FL_params.local_epoch):
    #             for batch_idx, (data, target) in enumerate(client_data_loader[0]):
    #                 data = data.to(device)
    #                 target = target.to(device)
    #                 optimizer = optim.SGD(m.parameters(), lr=FL_params.local_lr)
    #
    #                 optimizer.zero_grad()
    #                 pred = m(data)
    #                 # criteria = nn.CrossEntropyLoss()
    #                 criteria = nn.MSELoss()  # 更换为 MSELoss
    #                 target = target_to_output_shape(target, pred.shape)
    #                 # target = target.view(-1, 1).expand_as(pred).float()
    #                 loss = criteria(pred, target)
    #                 # loss = F.cross_entropy(pred, target)  # 计算交叉熵损失
    #                 loss.backward()
    #                 optimizer.step()
    #         client_models.append(m)
    #
    #     # client_models.to(device)
    #
    #     if ((FL_params.if_unlearning) and (
    #             FL_params.forget_client_idx in range(FL_params.N_client)) and not FL_params.if_sample_unlearning):
    #         client_models.pop(FL_params.forget_client_idx)
    #         print('Unlearn a client')
    #         return client_models
    #     else:
    #         return client_models
    # else:
    # print(global_model)
    global_model.to(FL_params.device)
    # device_cpu = torch.device("cpu")
    if (FL_params.data_name == "shakespeare"):
        text, data, string2integer, integer2string, vocab_size, chars = load_data("data/shakespeare.txt")
        clients_data, train_data, val_data = split_data(data, num_clients=3)
        model = train_model(chars, clients_data, train_data, val_data).to(FL_params.device)
        decode = lambda l: ''.join([integer2string[i] for i in l])
        generated_text = generate_text(model, decode)
        # print(generated_text)
        client_models = []
        for client_idx in range(FL_params.N_client):
            client_models.append(model)
        return client_models
    else:

        client_models = []
        client_sgds = []
        for ii in range(FL_params.N_client):
            client_models.append(copy.deepcopy(global_model))
            client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

        for client_idx in range(FL_params.N_client):
            model = client_models[client_idx]  # .to(device)

            if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or (
                    (FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
                continue
            # if((FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx)):
            #     continue
            # print(30*'-')
            # print("Now training Client No.{}  ".format(client_idx))

            # model = client_models[client_idx]
            optimizer = client_sgds[client_idx]

            # model
            model.train()

            # local training
            for local_epoch in range(FL_params.local_epoch):
                for batch_idx, (data, target) in enumerate(client_data_loader[client_idx]):
                    data = data.to(FL_params.device)
                    target = target.to(FL_params.device)
                    optimizer = optim.SGD(model.parameters(), lr=FL_params.local_lr)

                    optimizer.zero_grad()
                    pred = model(data)
                    # criteria = nn.CrossEntropyLoss()
                    criteria = nn.MSELoss()  # 更换为 MSELoss
                    target = target_to_output_shape(target, pred.shape)
                    # target = target.view(-1, 1).expand_as(pred).float()
                    loss = criteria(pred, target)
                    # loss = F.cross_entropy(pred, target)  # 计算交叉熵损失
                    loss.backward()
                    optimizer.step()
            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader, FL_params)

            # if (FL_params.if_sample_unlearning):
            #     for client_idx in FL_params.selected_K_group:
            #         model.to(device_cpu)
            #         client_models[client_idx] = model
            # else:
            model.to(FL_params.device)
            client_models[client_idx] = model

            if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx))):
                # 只有retrian 需要丢弃client 模型；如果不是在retrain的话，就不需要丢弃模型
                # Only retrian needs to discard the Client model;If it's not in Retrain, there's no need to discard the model
                client_models.pop(FL_params.forget_client_idx)
                return client_models
            elif ((FL_params.if_unlearning) and (
                    FL_params.forget_client_idx in range(FL_params.N_client)) and not FL_params.if_sample_unlearning):
                client_models.pop(FL_params.forget_client_idx)
                print('Unlearn a client')
                return client_models
            else:
                return client_models


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


def save_state(model, best_acc):
    print("==> Saving model ...")
    state = {
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
    }
    state_copy = state["state_dict"].copy()
    for key in state_copy.keys():
        if "module" in key:
            state["state_dict"][key.replace("module.", "")] = state["state_dict"].pop(
                key
            )
    return state


def FL_Finetuned(init_global_model, client_data_loaders, test_loader, FL_params):
    global_model = init_global_model
    epoch_acc = []

    for epoch in range(FL_params.finetune_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        global_model = fedavg(client_models)
        print("Finetune Global Federated Learning epoch = {}".format(epoch))

        (val_acc, test_loss) = test(global_model, test_loader, FL_params)
        (train_acc, train_loss) = test(global_model, client_data_loaders[-1], FL_params)

        epoch_acc.append((epoch, val_acc))

    epochs, accuracies = zip(*epoch_acc)

    return global_model, val_acc, epoch, test_loss, train_acc, train_loss


def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    if (FL_params.if_retrain == False):
        raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')
    # if (FL_params.forget_client_idx not in range(FL_params.N_client)):
    #     raise ValueError(
    #         'FL_params.forget_client_idx should be in [{}], if you want to use standard FL train with forget the certain client dataset.'.format(
    #             range(FL_params.N_client)))
    print('\n')
    print(5 * "#" + "  Federated Retraining Start  " + 5 * "#")
    # std_time = time.time()
    # print("Federated Retrain with Forget Client NO.{}".format(FL_params.forget_client_idx))
    retrain_GMs = list()
    all_client_models = list()
    retrain_GMs.append(copy.deepcopy(init_global_model))

    global_model = init_global_model
    project_dir = Path(__file__).resolve().parent
    save_info = project_dir / 'ckpt' / 'retrained'  # / FL_params.model_name

    epoch_acc = []

    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
        # 聚合，更新全局模型的参数，会应用到下一轮训练当中
        print("Global Federated Learning epoch = {}".format(epoch))
        global_model = fedavg(client_models)
        (val_acc, test_loss) = test(global_model, test_loader, FL_params)
        retrain_GMs.append(copy.deepcopy(global_model))
        all_client_models += client_models
        epoch_acc.append((epoch, val_acc))
        save_flag = save_net(global_model, val_acc, FL_params.save_re_acc, save_info, epoch)
        if save_flag:
            break
    epochs, accuracies = zip(*epoch_acc)

    # 将数据存储到Excel表中
    # print('Retrain Epoch', epochs, 'Retrain Accuracy', accuracies)
    data = {'Epoch': epochs, 'Accuracy': accuracies}
    df = pd.DataFrame(data)
    # excel_file = FL_params.data_name + '_' + FL_params.model_name + '_retrained.xlsx'
    # df.to_excel('./excel/' + excel_file, index=False)

    print(5 * "#" + "  Federated Retraining End  " + 5 * "#")
    return retrain_GMs


"""
Function：
Test the performance of the model on the test set
"""


def test(net, testloader, FL_params):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    if FL_params.data_name == "shakespeare":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # num_epochs = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练循环
        for epoch in range(FL_params.local_epoch):
            net.to(device)
            net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for texts, labels in testloader:
                    texts = texts.to(device)
                    labels = labels.to(device, dtype=torch.float)

                    outputs = net(texts)
                    predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch + 1}/{FL_params.local_epoch}], Accuracy: {100 * correct / total:.2f}%')
    else:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        net.to(device)

        net.eval()
        test_loss = 0
        test_acc = 0
        correct = 0
        total = 0

        with torch.no_grad():
            # for inputs, targets in enumerate(testloader):
            kwargs = {'num_workers': 1, 'pin_memory': True} if FL_params.cuda_state else {}

            for batch_idx, (inputs, targets) in enumerate(testloader):
                # inputs = inputs.to(device)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                test_acc += accuracy_score(predicted.cpu(), targets.cpu())
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        test_acc = test_acc / np.ceil(len(testloader.dataset) / testloader.batch_size)
        val_loss = test_loss / num_val_steps
        print("Test Loss=%.4f, Test accuracy=%.4f" % (val_loss / (num_val_steps), val_acc))
        # print('Test set: Average loss: {:.8f}'.format(val_loss))
        # print('Test set: Average acc:  {:.4f}'.format(val_acc))

    return (val_acc, val_loss)


def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    """


    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model

    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()  # Model Params： oldCM - oldGM_t
    new_param_update = dict()  # Model Params： newCM - newGM_t

    new_global_model_state = global_model_after_forget.state_dict()  # newGM_t

    return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    assert len(old_client_models) == len(new_client_models)

    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]

        return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]

        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
        new_param_update[layer] /= (ii + 1)  # Model Params： newCM

        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[
            layer]  # 参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[
            layer]  # 参数： newCM - newGM_t

        step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / torch.norm(
            new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)

    return_global_model.load_state_dict(return_model_state)

    return return_global_model


"""
Function：
FedAvg
"""


def fedavg(local_models):
    # def fedavg(local_models, local_model_weights=None):
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
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model
    # avg_state_dict = deepcopy(client_models[0].state_dict())
    #     for k in avg_state_dict.keys():
    #         for client_model in client_models[1:]:
    #             avg_state_dict[k] += client_model.state_dict()[k]
    #         avg_state_dict[k] = torch.div(avg_state_dict[k], len(client_models))
    #     return avg_state_dict
