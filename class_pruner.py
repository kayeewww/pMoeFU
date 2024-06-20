import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocess import data_set
from fedbabygpt import load_data, split_data, train_model, generate_text, BabyGPTmodel, GPTConfig, get_dataloaders
from sklearn.feature_extraction.text import TfidfVectorizer

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
                feature_iit[name] = feature.to(device)#.cpu()
            else:
                feature_iit[name] = torch.cat([feature_iit[name], feature.to(device)], 1)#.cpu()], 1)

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
        inputs, labels = inputs.to(device), targets.to(device)
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
def calculate_cp(features: dict, classes: list, dataset: str, coe: int, unlearn_client: int,device):
    # print('lens+++',len(classes)) #64
    features_class_wise = {}
    tf_idf_map = {}
    if dataset == 'cifar10'or dataset == 'mnist':
        class_num = 10
    if dataset == 'cifar100':
        class_num = 100
    if dataset == 'purchase'or dataset == 'adult':
        class_num = 2
    if dataset == 'shakespeare':
        class_num = 26
    list_classes_address = []
    # 把对应的类下的index存起来
    for z in range(class_num):
        address_index = [x for x in range(len(classes)) if classes[x] == z]
        list_classes_address.append([z, address_index])
    dict_address = dict(list_classes_address) #类别和对应的样本索引进行映射
    # features.to(device)
    for fea in features:
        # print('fea',fea)
        '''Class-wise Activation'''
        #创建了一个全零的张量，class_num是类别数量，features[fea].shape[0]是特征的维度大小。
        class_wise_features = torch.zeros(class_num, features[fea].shape[0], device=device)
        image_wise_features = features[fea].transpose(0, 1).to(device)
        # print(
            # f"Feature: {fea}, class_wise_features device: {class_wise_features.device}, image_wise_features device: {image_wise_features.device}")

        for i, v in dict_address.items():
            for j in v:
                class_wise_features[i] += image_wise_features[j]
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)

        features_class_wise[fea] = class_wise_features.transpose(0, 1)
        #将计算得到的类别特征向量存储在features_class_wise字典中，键为特征名称。
        # print(features_class_wise[fea].shape)

        calc_tf_idf(features_class_wise[fea], fea, coe=coe, unlearn_client=unlearn_client, tf_idf_map=tf_idf_map)

    return tf_idf_map

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

def Class_pruner(net, FL_params):

    models = []
    trainset, testset = data_set(FL_params.data_name)
    loaders = []

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=False)
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    for _ in range(FL_params.N_client):

        net.to(device)
        models.append(net)
        loaders.append(train_loader)
    tf_idf = []
    for m, l in zip(models, loaders):
        features, classes = acculumate_feature(m, l, stop=10)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 计算TF-IDF
        features = {k: v.to(device) for k, v in features.items()}  # 确保 features 在 GPU 上
        # print(f"Features and classes collected. Device of features: {[v.device for v in features.values()]}")

        tf_idf_map = calculate_cp(features, classes, dataset='cifar10', coe=1, unlearn_client=0,device=device)
        tf_idf.append(tf_idf_map)
    least_important_clients = select_least_important_clients(tf_idf, num_clients_to_select=FL_params.K)

    print("single tfidf Selected least important clients:", least_important_clients)
    return least_important_clients, tf_idf