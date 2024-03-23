import time
from pathlib import Path

import torch
import torchvision
import numpy as np
from torchvision import transforms

from FL_base import test, FL_Train, FL_Finetuned
from Fed_Unlearn_main import Arguments
from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity, TFIDFPruner
from model_initiation import Net_cifar10
from torch.utils.data import Dataset, TensorDataset, DataLoader


def load_model_pytorch(model, load_model, model_name):
    # print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    # 这段代码检查模型和加载的状态字典的键是否需要匹配
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    # 这段代码处理特定情况下的VGG模型。它将加载的状态字典中的键中的'features.'替换为'features'，以及将'classifier.'替换为'classifier'
    if model_name == "vgg":
        from collections import OrderedDict

        load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            # print(key, model.state_dict()[key].shape)
        # print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            # print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, False)


def generate(dataset, list_classes: list):
    labels = []
    for label_id in list_classes:
        labels.append(list(dataset.classes)[int(label_id)])
    # print(labels)

    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)
    return sub_dataset


def Class_pruner(FL_params):
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / 'ckpt' / FL_params.model_name / FL_params.model_file
    pruned_save_info = project_dir / 'ckpt' / 'pruned' / FL_params.model_name
    finetuned_save_info = project_dir / 'ckpt' / 'finetuned' / FL_params.model_name
    if FL_params.data_name == 'cifar10':
        '''load data and model'''
        mean = [125.31 / 255, 122.95 / 255, 113.87 / 255]
        std = [63.0 / 255, 62.09 / 255, 66.70 / 255]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR10(root=project_dir / 'data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=project_dir / 'data', train=False, download=True,
                                               transform=transform_test)

        net = Net_cifar10(FL_params.model_name)
        net.cuda()
        load_model_pytorch(net, model_path, FL_params.model_name)

        total_classes = 10  # [0-9]

    train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)

    '''pre-processing'''
    feature_iit, classes = acculumate_feature(net, train_all_loader, 1)
    tf_idf_map = calculate_cp(feature_iit, classes, FL_params.data_name, 0, unlearn_class=FL_params.unlearn_class)
    threshold = get_threshold_by_sparsity(tf_idf_map, FL_params.sparsity)
    print('threshold', threshold)

    '''test before pruning'''
    list_allclasses = list(range(total_classes))
    unlearn_listclass = [FL_params.unlearn_class]
    list_allclasses.remove(FL_params.unlearn_class)  # rest classes
    unlearn_testdata = generate(testset, unlearn_listclass)
    rest_testdata = generate(testset, list_allclasses)
    print(len(unlearn_testdata), len(rest_testdata))
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=FL_params.test_batch_size,
                                                     shuffle=False)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=FL_params.test_batch_size,
                                                  shuffle=False)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)

    device_cpu = torch.device("cpu")
    net.to(device_cpu)
    test(net, unlearn_testloader)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(net, rest_testloader)
    print('*' * 40)

    '''pruning'''
    cp_config = {"threshold": threshold, "map": tf_idf_map}
    config_list = [{
        'sparsity': FL_params.sparsity,
        'op_types': ['Conv2d']
    }]
    pruner = TFIDFPruner(net, config_list, cp_config=cp_config)
    pruner.compress()
    pruned_model_path = pruned_save_info / ('seed_' +
                                            time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) +
                                            '_model.pth')
    pruned_mask_path = pruned_save_info / ('seed_' +
                                           time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) +
                                           '_mask.pth')
    pruner.export_model(pruned_model_path, pruned_mask_path)

    pruned_net = Net_cifar10(FL_params.model_name)
    pruned_net.cuda()
    load_model_pytorch(pruned_net, pruned_model_path, FL_params.model_name)

    '''test after pruning'''
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    pruned_net.to(device_cpu)
    test(pruned_net, unlearn_testloader)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(pruned_net, rest_testloader)
    print('*' * 40)

    # return #for test

    '''fine tuning'''
    rest_traindata = generate(trainset, list_allclasses)
    print(len(rest_traindata))

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    # 找到要剔除类别的索引
    indices_to_remove = [i for i, label in enumerate(trainset.targets) if label == FL_params.unlearn_class]

    # 从训练集和测试集中删除对应类别的样本
    trainset.data = [image for i, image in enumerate(trainset.data) if i not in indices_to_remove]
    trainset.targets = [label for i, label in enumerate(trainset.targets) if i not in indices_to_remove]
    testset.data = [image for i, image in enumerate(testset.data) if i not in indices_to_remove]
    testset.targets = [label for i, label in enumerate(testset.targets) if i not in indices_to_remove]

    # 创建一个长度为 N_total_client-1 的列表，每个元素都是每个客户端数据集的样本数量
    split_index = [int(trainset.__len__() / FL_params.N_total_client)] * (FL_params.N_total_client - 1)
    # 计算最后一个客户端数据集的样本数量,即为所有的样本数减去前面计算的所有样本数量
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_params.N_total_client) * (
                FL_params.N_total_client - 1)))
    # split_index：将上述计算得到的客户端数据集的样本数量组成的列表
    # 用random_split将训练集划分为给定数量的数据集
    rest_client_data = torch.utils.data.random_split(trainset, split_index)

    # rest_client_loaders = torch.utils.data.DataLoader(rest_traindata, batch_size=FL_params.local_batch_size,shuffle=False)

    all_rest_client_loaders = []
    for ii in range(FL_params.N_total_client):
        all_rest_client_loaders.append(
            DataLoader(rest_client_data[ii], batch_size=FL_params.local_batch_size, shuffle=True, **kwargs))

    rest_testloader = torch.utils.data.DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)

    # 从100个客户端中随机抽取25个作为实验对象
    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    rest_client_loaders = list()
    for idx in selected_clients:
        rest_client_loaders.append(all_rest_client_loaders[idx])

    print(5 * "#" + "  Federated Fine-tuning Start" + 5 * "#")
    train_model, train_acc, train_epoch = FL_Finetuned(pruned_net, rest_client_loaders, rest_testloader, FL_params)
    print("train acc:%.4f" % train_acc)
    print("train epoch:%d" % train_epoch)
    print(5 * "#" + "  Federated Fine-tuning End" + 5 * "#")

    # train(pruned_net, epochs=args.epochs, lr=args.lr, train_loader=rest_trainloader,
    #       test_loader=rest_testloader, save_info=finetuned_save_info, save_acc=args.save_acc, seed=args.seed,
    #       label_smoothing=args.label_smoothing, warmup_step=args.warmup_step, warm_lr=args.warm_lr)


#
# '''test after fine-tuning'''
# print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
# test(pruned_net, unlearn_testloader)
# print('*' * 40)
# print('*' * 5 + 'testing in rest_data' + '*' * 15)
# test(pruned_net, rest_testloader)
# print('*' * 40)
#
# print('finished')


if __name__ == '__main__':
    FL_params = Arguments()
    Class_pruner(FL_params)
