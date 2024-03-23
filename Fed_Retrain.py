from pathlib import Path

import torch
import torchvision
import numpy as np
from torchvision import transforms,datasets

from FL_base import FL_Finetuned, test, FL_Retrain
from Fed_Unlearn_main import Arguments
from Fed_pruner import load_model_pytorch, generate
from model_initiation import Net_cifar10, model_init
from torch.utils.data import DataLoader


def Retraining(FL_params):
    project_dir = Path(__file__).resolve().parent
    save_info = project_dir / 'ckpt' / 'retrained' / FL_params.model_name

    if FL_params.data_name == 'cifar10':
        '''load data and model'''
        mean = [125.31 / 255, 122.95 / 255, 113.87 / 255]
        std = [63.0 / 255, 62.09 / 255, 66.70 / 255]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    total_classes = 10  # [0-9]
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                            transform=transform_test)

    # net = Net_cifar10(FL_params.model_name)
    # net.cuda()
    net = model_init(FL_params.data_name, FL_params.model_name)

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    # list_allclasses = list(range(total_classes))
    # unlearn_listclass = [FL_params.unlearn_class]
    # list_allclasses.remove(FL_params.unlearn_class)  # rest classes
    # rest_traindata = generate(trainset, list_allclasses)
    # rest_testdata = generate(testset, list_allclasses)
    # unlearn_testdata = generate(testset, unlearn_listclass)
    # print(len(rest_traindata), len(rest_testdata), len(unlearn_testdata))
    # rest_trainloader = torch.utils.data.DataLoader(rest_traindata, batch_size=FL_params.local_batch_size,
    #                                                shuffle=False)
    # rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=FL_params.test_batch_size,
    #                                               shuffle=True, **kwargs)
    # unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=FL_params.test_batch_size,
    #                                                 shuffle=True, **kwargs)



    # 找到要剔除类别的索引
    indices_to_remove = [i for i, label in enumerate(trainset.targets) if label == FL_params.unlearn_class]

    # 从训练集和测试集中删除对应类别的样本
    trainset.data = [image for i, image in enumerate(trainset.data) if i not in indices_to_remove]
    trainset.targets = [label for i, label in enumerate(trainset.targets) if i not in indices_to_remove]
    testset.data = [image for i, image in enumerate(testset.data) if i not in indices_to_remove]
    testset.targets = [label for i, label in enumerate(testset.targets) if i not in indices_to_remove]



    # 输出剔除特定类别后的训练集大小
    print("剔除类别后的训练集大小:", len(trainset))

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

    print(5 * "#" + "  Federated Retraining Start" + 5 * "#")
    train_model, train_acc, train_epoch = FL_Retrain(net, rest_client_loaders, rest_testloader, FL_params)
    print("train acc:%.4f" % train_acc)
    print("train epoch:%d" % train_epoch)
    print(5 * "#" + "  Federated Retraining End" + 5 * "#")
    '''training'''
    # train(net, epochs=args.epochs, lr=args.lr, train_loader=rest_trainloader,
    #       test_loader=rest_testloader, save_info=save_info, save_acc=args.save_acc, seed=args.seed,
    #       label_smoothing=args.label_smoothing, warmup_step=args.warmup_step, warm_lr=args.warm_lr)

    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    device_cpu = torch.device("cpu")
    net.to(device_cpu)
    print('*' * 40)

    print('finished')


if __name__ == '__main__':
    FL_params = Arguments()
    Retraining(FL_params)