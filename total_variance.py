import copy
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
# from Fed_Unlearn_base import
from FL_base import FL_Finetuned,fedavg, global_train_once, unlearning_step_once
from class_pruner import Class_pruner
from moe import MoE
import data_preprocess
from data_preprocess import model_init
from sklearn.metrics import accuracy_score

from fedbabygpt import load_data, split_data, train_model, generate_text, BabyGPTmodel, GPTConfig, get_client_batch
from expert_model import em_init, emloader_init

import torch
def setExpers(global_model, client_data_loaders, FL_params):
    print('#'*8, 'Different models and dataset experts','#'*8)
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # num_experts = FL_params.N_client

    # 创建 experts 的 ModuleList，其中每个专家使用不同的模型和数据集
    experts = nn.ModuleList()
    input_sizes = []
    output_sizes = []
    # 初始化每个专家的模型
    for i in range(len(FL_params.datasets)):
        dataset_name = FL_params.datasets[i]
        model, input_size, output_size = em_init(dataset_name, device)
        experts.append(model)
        # input_sizes.append(input_size[0] * input_size[1] * input_size[2])
        input_sizes.append(input_size)
        output_sizes.append(output_size)
        data_loader = emloader_init(dataset_name, FL_params)
        client_data_loaders.append(data_loader)


    # selected_client, tf_idf_scores = Class_pruner(global_model, FL_params)
    tf_idf_scores = [5, 2]
    moe_model = MoE(
        experts=experts,
        num_experts=FL_params.N_client,
        input_sizes=input_sizes,
        output_size=output_sizes,
        tf_idf_scores=tf_idf_scores,
        forget_client_idx=FL_params.forget_client_idx,
        k=FL_params.forget_clients_num,
        noisy_gating=True
    )
    moe_model = moe_model.to(device)

    # 假设 expert_idx 是当前选择的专家模型索引
    # for expert in range(FL_params.N_client):
    dataiter = iter(client_data_loaders[-1])
    images, labels = next(dataiter)

    # 根据数据集调整输入形状
    dataset_name = FL_params.datasets[-1]
    if dataset_name == 'cifar10':
        images = images.view(-1, 3, 32, 32).to(device)
        # expert_outputs = [self.experts[i](expert_inputs[i].view(-1, 3, 32, 32)) for i in range(self.num_experts)]
    elif dataset_name == 'mnist':
        images = images.view(-1, 1, 28, 28).to(device)
    elif dataset_name == 'shakespeare':
        images = images.to(device)
    expert_idx = -1  # 这里假设使用最后一个专家模型
    outputs, loss = moe_model(images, expert_idx)
    print('One expert: ', outputs, loss)


    return experts
# Algorithm: Client-level Unlearning for FATS
def client_level_unlearning(global_model, old_client_models, client_data_loaders, test_loader, FL_params):
    """
    Unlearn an entire client from the federated learning process.
    t_u: unlearning time step
    k_u: number of target to unlearn
    """
    # Initialize global model parameters
    old_global_models = copy.deepcopy(global_model)
    old_client_models = copy.deepcopy(old_client_models)
    new_GMs = list()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    for epoch in range(FL_params.global_epoch):
        if (epoch == 0):
            continue
        # print("Client-level Federated Unlearning Global Epoch  = {}".format(epoch))
        old_global_model = old_global_models[epoch].to(device)
        global_model = old_global_model.to(device)
        global_model.train()
        new_GMs.append(global_model)


    # Time steps
    client_states = {k: {'model': copy.deepcopy(global_model).to(device)} for k in range(FL_params.N_client)}

    split_clients = np.array_split(list(client_states),
                                   np.arange(FL_params.K, len(client_states), FL_params.K))

    # flat_clients = [item for sublist in split_clients for item in sublist]
    print('split_clients', split_clients)

    selected_client, tf_idf_scores = Class_pruner(global_model, FL_params)
    experts = setExpers(global_model, client_data_loaders, FL_params)
    if FL_params.data_name=='cifar10':
        num_experts = FL_params.N_client
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        experts = nn.ModuleList([data_preprocess.model_init('cifar10', device) for i in range(num_experts)])
        moe_model = MoE(input_size=3 * 32 * 32, output_size=10, experts=experts, num_experts=num_experts, hidden_size=100,
                        tf_idf_scores=tf_idf_scores, forget_client_idx=FL_params.forget_client_idx,
                        k=FL_params.forget_clients_num, noisy_gating=True)
        # print(tf_idf_scores)
        moe_model = moe_model.to(device)
        dataiter = iter(client_data_loaders[-1])
        images, labels = next(dataiter)

        # 将 CIFAR-10 数据集的图像转换为模型输入格式
        images = images.to(device)
        # outputs = moe_model(images)
        outputs = moe_model(images.view(-1, 3 * 32 * 32))
    elif FL_params.data_name=='mnist':
        num_experts = FL_params.N_client
        experts = nn.ModuleList([data_preprocess.model_init('mnist', device) for i in range(num_experts)])
        moe_model = MoE(input_size=1 * 28 * 28, output_size=10, experts=experts, num_experts=10, hidden_size=100,
                        tf_idf_scores=tf_idf_scores, forget_client_idx=FL_params.forget_client_idx,
                        k=FL_params.forget_clients_num, noisy_gating=True)
        # print(tf_idf_scores)
        moe_model = moe_model.to(device)

        dataiter = iter(client_data_loaders[-1])
        images, labels = next(dataiter)

        # 将 MNIST 数据集的图像转换为模型输入格式
        images = images.to(device)
        outputs = moe_model(images.view(-1, 1 * 28 * 28))

    elif FL_params.data_name == 'shakespeare':
        text, data, string2integer, integer2string, vocab_size, chars = load_data("data/shakespeare.txt")

        num_experts = FL_params.N_client
        block_size = 4
        experts = nn.ModuleList([data_preprocess.model_init('shakespeare', device) for i in range(num_experts)])
        moe_model = MoE(input_size=block_size, output_size=block_size, experts=experts, num_experts=num_experts, hidden_size=100,
                        tf_idf_scores=tf_idf_scores, forget_client_idx=FL_params.forget_client_idx,
                        k=FL_params.forget_clients_num, noisy_gating=True)
        # print(tf_idf_scores)
        moe_model = moe_model.to(device)

        for images, labels in client_data_loaders:
            outputs = moe_model(images)
            # 其他逻辑

    # 打印输出和选择的 client index

    print("Forget Client Index:", moe_model.forget_client_idx)
    FL_params.forget_client_idx = moe_model.forget_client_idx
    print('Client chosen by MoE: ', FL_params.forget_client_idx)
    print("Initial model type", type(moe_model), moe_model)

    client_models = copy.deepcopy(moe_model)

    updated_global_models = list()
    client_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # if 1:#k_u in selected_clients:
    temp=old_client_models
    if type(FL_params.forget_client_idx) == int:
        temp.pop(FL_params.forget_client_idx)
    else:
        for k_client in FL_params.forget_client_idx:
            temp.pop(k_client)
    # for k_client in FL_params.forget_client_idx:
    #     temp.pop(k_client)
        # remain_client_list = client_list.pop(k_client)
    # remain_client_list=[]
    client_set=set(client_list)
    forget_set=set(FL_params.forget_client_idx)
    remain_client_list = list(client_set-forget_set)
    print('After Removing', remain_client_list)
    old_client_models.append(temp)

    new_c_data_loaders = [dl for i, dl in enumerate(client_data_loaders) if i != FL_params.forget_client_idx]
    new_client_data_loaders = []
    for dl in new_c_data_loaders:
        new_client_data_loader = DataLoader(dl.dataset, batch_size=FL_params.b, shuffle=False)
        new_client_data_loaders.append(new_client_data_loader)

    local_client_models = list()

    # for client in FL_params.selected_K_group:
    for client in remain_client_list:
        # data_loader = new_client_data_loaders[client]
        local_model = client_models.to(device)  # old_client_models[client]

        for itr in range(FL_params.local_epoch):
            # local_model = local_model1[itr]
            # sampled_dataloader = np.random.choice(list(data_loader), FL_params.b, replace=True)
            for inputs, targets in new_client_data_loaders[itr]:
                inputs, targets = inputs.to(device), targets.to(device)
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
        fedavg_global_model = fedavg(local_client_models).to(device)
        unlearn_global_model.append(fedavg_global_model)

    updated_global_models.append(unlearn_global_model)

    print("Updated global", updated_global_models, type(updated_global_models), len(updated_global_models),
          type(updated_global_models[0]))

    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch * FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)
    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    # FL_params.global_epoch = CM_intv.shape[0]

    # print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(0, 1):  # global_epoch):
        if (epoch == 0):
            continue
        print("Client-level Unlearning Local Calibration Training epoch  = {}".format(epoch))
        global_model = updated_global_models[epoch][0].experts
        # global_model = updated_global_models[epoch]

        new_client_models = global_train_once(global_model, new_client_data_loaders, test_loader, FL_params)

        new_GM = unlearning_step_once(client_models[epoch], new_client_models, client_models[epoch + 1],
                                      global_model)

        updated_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch

    '''fine tuning'''

    finetuned_global_models = updated_global_models[-1]
    tv_stability = calculate_total_variance_stability(global_model, finetuned_global_models[-1],remain_client_list,
                                                      new_client_data_loaders, test_loader)
    prune_model_save_dir = "ckpt/prune"
    print('TV stability: ', tv_stability)
    iteration_count = 0
    if tv_stability > FL_params.tv_stability_threshold:
        print(5 * "#" + "  Federated Fine-tuning Start" + 5 * "#")
        finetuned_epoch=0
        while tv_stability > FL_params.tv_stability_threshold and FL_params.rouc < 1.0:
            finetuned_global_model = finetuned_global_models[-1]
            iteration_count += 1
            FL_params.rouc += 0.05
            FL_params.finetune_epoch = 40
            finetuned_global_model, test_acc, train_epoch, test_loss, train_acc, train_loss = FL_Finetuned(
                finetuned_global_model, new_client_data_loaders, test_loader, FL_params)
            tv_stability = calculate_total_variance_stability(global_model, finetuned_global_model,remain_client_list,
                                                              new_client_data_loaders, test_loader)
            # 保存每个专家模型的状态
            for i, expert_model in enumerate(finetuned_global_model.experts):
                prune_model_path = os.path.join(prune_model_save_dir, f"pruned_model_expert_{i}_iter_{iteration_count}_{FL_params.K}K_{FL_params.M}_forget{FL_params.forget_clients_num}.pth")
                torch.save(expert_model.state_dict(), prune_model_path)

            # 打印当前迭代次数和 TV 稳定性
            print(f"Iteration {iteration_count}: TV Stability = {tv_stability}, rouc = {FL_params.rouc}")

        print(f"Final rouc value: {FL_params.rouc}")
        print(f'We have {iteration_count} epochs fine-tuning')
        print(f"Final TV stability: {tv_stability}")
    else:
        print('Here basic finetuning:')
        FL_params.finetune_epoch = 20
        finetuned_global_model, test_acc, train_epoch, test_loss, train_acc, train_loss = FL_Finetuned(
            finetuned_global_models[-1], new_client_data_loaders, test_loader, FL_params)

        print("Fine-tuning train acc:%.4f" % train_acc)
        print("Fine-tuning train epoch:%d" % train_epoch)
        print(5 * "#" + "  Federated Fine-tuning End" + 5 * "#")
    updated_global_models.append(finetuned_global_model)

    return updated_global_models, train_acc, train_loss, test_acc, test_loss

def calculate_total_variance_stability(global_model, new_global_model, remain_client_list, new_client_data_loaders, test_loader):
    # 实现总方差稳定性的计算
    # 通过比较模型在遗忘前后在测试集上的性能变化来计算总方差稳定性
    # print(f"Global model device: {next(global_model.parameters()).device}")
    # print(f"New GM device: {next(new_GM.parameters()).device}")

    old_model_performance = evaluate_model(global_model, remain_client_list, new_client_data_loaders, test_loader)
    new_model_performance = evaluate_model(new_global_model, remain_client_list, new_client_data_loaders, test_loader)
    tv_stability = np.abs(old_model_performance - new_model_performance)
    return tv_stability

def evaluate_model(model, remain_client_list, new_client_data_loaders, test_loader):
    # device=torch.device("cuda:1" if torch.cuda.is_available() else "")
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    performance = 0
    with torch.no_grad():
        for client in remain_client_list:
            client_performance = 0
            num_batches = 0
            for inputs, targets in new_client_data_loaders[client]:
                inputs.to(device)
                targets.to(device)
                outputs = model(inputs)
                performance += accuracy_score(targets.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy())
                num_batches += 1
                # performance += accuracy_score(targets.to(device), outputs.argmax(dim=1).to(device))
    performance /= len(remain_client_list)
    performance /=100
    return performance