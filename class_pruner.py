import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocess import data_set
from fedbabygpt import load_data, split_data, train_model, generate_text, BabyGPTmodel, GPTConfig, get_dataloaders
from sklearn.feature_extraction.text import TfidfVectorizer
import copy

# def compute_client_tfidf(clients_data, block_size):
#     tf_idf_list = []
#     for client, data in clients_data.items():
#         train_data_str = data['train']
#         print(f"Client {client} train_data_str: {train_data_str[:100]}")  # 打印前100个字符进行检查
#         vectorizer = TfidfVectorizer(analyzer='char', max_features=block_size)
#         tf_matrix = vectorizer.fit_transform([train_data_str])
#         tf_idf_map = {i: torch.tensor(tf_matrix[0, i]) for i in range(tf_matrix.shape[1])}
#         tf_idf_list.append(tf_idf_map)
#     return tf_idf_list

# %%
def acculumate_feature(model, loader, stop: int):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    model.to(device)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    features = {}
    classes = []
    all_features = []
    all_classes = []

    def hook_func(m, x, y, name, feature_iit):
        f = F.relu(y)
        if f.size()[3] != 0:
            # feature.to(device)
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

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        model.eval()
        # inputs, labels = inputs.to(device), targets.to(device)
        classes.extend(targets.numpy())
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_features.append(outputs)
            all_classes.append(targets)

    all_features = torch.cat(all_features, dim=0)
    all_classes = torch.cat(all_classes, dim=0)
    [k.remove() for k in handler_list]

    return features, classes

# 计算特征的TF-IDF（Term Frequency-Inverse Document Frequency），并将结果存储在tf_idf_map字典中
#tf_idf_map = calculate_cp(feature_iit, client_list, FL_params.data_name, 0, FL_params.forget_client_idx)
def calculate_cp(features, classes, dataset, coe, unlearn_client, device):
    # Ensure classes are in long format for indexing
    classes = classes.long()

    # Assuming features is a tensor of shape [num_samples, feature_dim]
    class_num = classes.max().item() + 1
    feature_dim = features.shape[1]  # Assuming features are of shape [num_samples, feature_dim]

    class_wise_features = torch.zeros(class_num, feature_dim, device=device)

    for fea in range(feature_dim):
        for cls in range(class_num):
            class_mask = (classes == cls).float().unsqueeze(1)
            class_wise_features[cls, fea] = torch.mean(features[:, fea].unsqueeze(1) * class_mask)

    tf_idf_map = calculate_tf_idf(class_wise_features)  # Your TF-IDF calculation logic here
    return tf_idf_map
def calculate_tf_idf(class_wise_features):
    """
    Calculate TF-IDF scores for the given class-wise features.

    Args:
    class_wise_features (torch.Tensor): Tensor of shape [class_num, feature_dim], containing the mean features for each class.

    Returns:
    torch.Tensor: Tensor of shape [class_num, feature_dim], containing the TF-IDF scores for each class and feature.
    """
    class_num, feature_dim = class_wise_features.shape

    # Calculate term frequency (TF)
    term_frequency = class_wise_features / (class_wise_features.sum(dim=1, keepdim=True) + 1e-10)

    # Calculate document frequency (DF)
    document_frequency = (class_wise_features > 0).sum(dim=0, dtype=torch.float)

    # Calculate inverse document frequency (IDF)
    idf = torch.log((class_num / (document_frequency + 1e-10)) + 1)

    # Calculate TF-IDF
    tf_idf = term_frequency * idf.unsqueeze(0)

    return tf_idf

# c - filters; n - classes
# feature = [c, n] ([64, 10])
def calc_tf_idf(feature, name: str, coe: int, unlearn_client: int, tf_idf_map: dict):
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0).to(feature.device)
    # 沿着第一个维度（通常是样本维度）的特征张量中每个特征在所有样本上的总和
    # print(feature_sum.shape) # ([10])
    balance_coe = np.log((feature.shape[0] / coe) * np.e) if coe else 1.0
    # print(feature.shape, name, coe)
    # 用于平衡各个特征之间的重要性，以防止某些特征过度强调。

    tf = (feature / sum_on_filters) * balance_coe
    # print(tf.shape) # ([64, 10])
    tf_unlearn_class = tf.transpose(0, 1)[unlearn_client]
    # print(tf_unlearn_class.shape)
    # tf_unlearn_class = tf[:, unlearn_class]
    # print(f"TF for unlearn class shape: {tf_unlearn_class.shape}")  # Expected to be [number_of_features]

    # calc idf for filters
    classes_quant = float(feature.shape[1])
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1).to(feature.device)
    # print(mean_on_classes.shape) # ([64, 1])
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor).to(feature.device)
    # print(inverse_on_classes.shape) # ([64])
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0)).to(feature.device)
    # print(idf.shape) # ([64])

    importance = tf_unlearn_class * idf
    # print(importance.shape) # ([64])
    tf_idf_map[name] = importance
    return tf_idf_map

def get_threshold_by_sparsity(mapper: dict, sparsity: float):
    assert 0 < sparsity < 1
    # print(len(mapper.values())) # 19
    tf_idf_array = torch.cat([v for v in mapper.values()], 0)
    # print(tf_idf_array.shape) # ([688])
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0] * (1 - sparsity)))[0].min()
    return threshold

def select_least_important_clients(tf_idf_list, num_clients_to_select):
    client_importance = {}

    for client_id, tf_idf_map in enumerate(tf_idf_list):
        client_importance[client_id] = 0
        for feature_importance in tf_idf_map.values():
            client_importance[client_id] += feature_importance.sum().item()

    # 按照重要性排序，选择重要性最低的客户端
    # print('client_importance', client_importance)
    sorted_clients = sorted(client_importance, key=client_importance.get)
    least_important_clients = sorted_clients[:num_clients_to_select]

    return least_important_clients

# def Class_pruner(net, train_loader, FL_params):
#
#     models = []
#     loaders = []
#     for _ in range(FL_params.N_client):
#         net.to(FL_params.device)
#         models.append(net)
#         loaders.append(train_loader)
#     tf_idf = []
#     for m, l in zip(models, loaders):
#         features, classes = acculumate_feature(m, l, stop=10)
#         # 计算TF-IDF
#         features = {k: v.to(FL_params.device) for k, v in features.items()}
#         # print(f"Features and classes collected. Device of features: {[v.device for v in features.values()]}")
#         tf_idf_map = calculate_cp(features, classes, dataset=FL_params.data_name, coe=1, unlearn_client=0,
#                                   device=FL_params.device)
#         tf_idf.append(tf_idf_map)
#     least_important_clients = select_least_important_clients(tf_idf, num_clients_to_select=FL_params.forget_clients_num)
#
#     print("TFIDF Selected least important clients:", least_important_clients)
#     return least_important_clients, tf_idf
def calculate_tfidf_scores(idxs_users, dataset_train, dict_users, net_locals, FL_params):
    tf_idf_scores = []
    for idx in idxs_users:
        client_indices = dict_users[idx]
        client_loader = [dataset_train[i] for i in client_indices]  # 获取该客户端的所有数据样本
        net_local = net_locals[idx]
        net_local.to(FL_params.device)
        net_local.eval()  # 设置模型为评估模式

        # Assuming client_loader is a list of tuples (img, target)
        features = []
        classes = []
        with torch.no_grad():  # 在评估模式下禁用梯度计算
            for img, target in client_loader:
                img = img.to(FL_params.device)
                feature = net_local(img.unsqueeze(0))  # Unsqueeze to add batch dimension
                features.append(feature)
                classes.append(target)

        features = torch.cat(features, dim=0)
        classes = torch.tensor(classes).to(FL_params.device)

        tf_idf_map = calculate_cp(features, classes, dataset=FL_params.data_name, coe=1, unlearn_client=0,
                                  device=FL_params.device)
        # tf_idf_scores.append(tf_idf_map)
        tf_idf_scores.append(tf_idf_map.mean().item())  # 取平均值并转换为标量

    return tf_idf_scores


def Class_pruner(net, client_loaders, FL_params):
    # models = []
    # loaders = []
    # for _ in range(FL_params.N_client):
    #     net.to(FL_params.device)
    #     models.append(copy.deepcopy(net))
    #     loaders.append(client_loaders[_])
    # print(net)
    # print(client_loaders)
    net_locals = net[0]['local']
    client_loaders_list = list(client_loaders.values())
    # tf_idf = calculate_tfidf_scores(range(len(net_locals)), client_loaders_list, dict(), net_locals, FL_params)

    tf_idf = []
    for m, l in zip(net_locals, client_loaders_list):
        # print('m_local:', m)
        m.to(FL_params.device)
        features, classes = acculumate_feature(m, l, stop=10)
        features = {k: v.to(FL_params.device) for k, v in features.items()}
        tf_idf_map = calculate_cp(features, classes, dataset=FL_params.data_name, coe=1, unlearn_client=0,
                                  device=FL_params.device)
        tf_idf.append(tf_idf_map)

    least_important_clients = select_least_important_clients(tf_idf, num_clients_to_select=FL_params.forget_clients_num)
    print("TFIDF Selected least important clients:", least_important_clients)
    return least_important_clients, tf_idf
