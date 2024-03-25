import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.version import cuda


# import L1FilterPruner, L1FilterPrunerMasker
# from nni import L1FilterPruner, L1FilterPrunerMasker
# from nni.compression.pruning import L1FilterPrunerMasker
# from nni.compression.pruning import L1FilterPruner

from nni.compression.pruning import L1NormPruner, L2NormPruner
import copy
# from nni.compression.pruning import structured_pruning


# %%
def acculumate_feature(model, loader, stop: int):
    if torch.cuda.is_available():
        model = model.cuda()
    features = {}
    classes = []

    def hook_func(m, x, y, name, feature_iit):
        # print(name, y.shape) # ([256, 64, 8, 8])
        '''ReLU'''
        f = F.relu(y)
        # f = y
        '''Average Pool'''
        feature = F.avg_pool2d(f, f.size()[3])
        # print(feature.shape) # ([256, 64, 1, 1])
        feature = feature.view(f.size()[0], -1)
        # print(feature.shape) # ([256, 64])
        feature = feature.transpose(0, 1)
        # print(feature.shape)
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)

    hook = functools.partial(hook_func, feature_iit=features)

    handler_list = []
    # 遍历模型的所有模块，对于每一个卷积层，注册一个前向钩子，用于在模型前向传播时提取特征，并将这些钩子的句柄存储在handler_list列表中。
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # if not isinstance(m, nn.Linear):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)

    # 在此循环中，对数据加载器进行迭代，获取每个批次的输入数据和目标标签，并通过模型进行前向传播以提取特征。当达到指定的stop批次时，停止迭代。
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        if batch_idx % (10) == 0:
            print('batch_idx', batch_idx)
        model.eval()
        classes.extend(targets.numpy())
        with torch.no_grad():
            model(inputs)
            # model(inputs.cuda())
    [k.remove() for k in handler_list]
    '''Image-wise Activation'''
    return features, classes


# 计算特征的TF-IDF（Term Frequency-Inverse Document Frequency），并将结果存储在tf_idf_map字典中
def calculate_cp(features: dict, classes: list, dataset: str, coe: int, unlearn_class: int, tf_idf_map: dict):
    # print('lens+++',len(classes)) #64
    features_class_wise = {}
    # tf_idf_map = {}
    if dataset == 'cifar10'or dataset == 'mnist':
        class_num = 10
    if dataset == 'cifar100':
        class_num = 100
    if dataset == 'purchase'or dataset == 'adult':
        class_num = 2
    list_classes_address = []
    # 把对应的类下的index存起来
    for z in range(class_num):
        address_index = [x for x in range(len(classes)) if classes[x] == z]
        list_classes_address.append([z, address_index])
    dict_address = dict(list_classes_address) #类别和对应的样本索引进行映射
    for fea in features:
        '''Class-wise Activation'''
        #创建了一个全零的张量，class_num是类别数量，features[fea].shape[0]是特征的维度大小。
        class_wise_features = torch.zeros(class_num, features[fea].shape[0])
        image_wise_features = features[fea].transpose(0, 1)
        for i, v in dict_address.items():
            for j in v:
                class_wise_features[i] += image_wise_features[j]
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)
        features_class_wise[fea] = class_wise_features.transpose(0, 1)
        #将计算得到的类别特征向量存储在features_class_wise字典中，键为特征名称。
        # print(features_class_wise[fea].shape) # ([64, 10])

        '''TF-IDF'''
        calc_tf_idf(features_class_wise[fea], fea, coe=coe,
                    unlearn_class=unlearn_class, tf_idf_map=tf_idf_map)
        # print(tf_idf_map[fea].shape)
    return tf_idf_map


# c - filters; n - classes
# feature = [c, n] ([64, 10])
def calc_tf_idf(feature, name: str, coe: int, unlearn_class: int, tf_idf_map: dict):
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0)
    # 沿着第一个维度（通常是样本维度）的特征张量中每个特征在所有样本上的总和
    # print(feature_sum.shape) # ([10])
    balance_coe = np.log((feature.shape[0] / coe) * np.e) if coe else 1.0
    # print(feature.shape, name, coe)
    # 用于平衡各个特征之间的重要性，以防止某些特征过度强调。

    tf = (feature / sum_on_filters) * balance_coe
    # print(tf.shape) # ([64, 10])
    tf_unlearn_class = tf.transpose(0, 1)[unlearn_class]
    # print(tf_unlearn_class.shape)

    # calc idf for filters
    classes_quant = float(feature.shape[1])
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1)
    # print(mean_on_classes.shape) # ([64, 1])
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor)
    # print(inverse_on_classes.shape) # ([64])
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0))
    # print(idf.shape) # ([64])

    importance = tf_unlearn_class * idf
    # print(importance.shape) # ([64])
    tf_idf_map[name] = importance
    return tf_idf_map


#根据给定的稀疏度（sparsity）从TF-IDF值的映射（mapper字典）中获取一个阈值，该阈值用于剪枝（pruning）操作
def get_threshold_by_sparsity(mapper: dict, sparsity: float):
    assert 0 < sparsity < 1
    # print(len(mapper.values())) # 19
    # 将mapper字典中所有特征的TF-IDF值拼接成一个一维张量tf_idf_array
    tf_idf_array = torch.cat([v for v in mapper.values()], 0)
    # print(tf_idf_array.shape) # ([688])
    # 找到TF-IDF值中按大小排序的前int(tf_idf_array.shape[0] * (1 - sparsity))个值，并取这些值中的最小值作为阈值
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0] * (1 - sparsity)))[0].min()
    threshold = float(threshold.item())
    return threshold


class TFIDFPruner(L1NormPruner):
    def __init__(self, model, config_list, cp_config: dict, pruning_algorithm='l1',
                 optimizer=None, **algo_kwargs):
        # model（待剪枝的模型）、config_list（剪枝配置列表）、
        # cp_config（剪枝配置，应该是一个字典）、pruning_algorithm（剪枝算法，默认为 'l1'）、
        # optimizer（优化器，默认为 None）以及其他参数通过 **algo_kwargs 传递。
        super().__init__(model, config_list, optimizer)
        # self.set_wrappers_attribute("if_calculated", False) #用于跟踪是否已经计算了TF-IDF值
        self.model = model
        self.masker = TFIDFMasker(model, self, threshold=cp_config["threshold"],
                                  tf_idf_map=cp_config["map"], **algo_kwargs)

    def update_masker(self, model, threshold, mapper):
        self.masker = TFIDFMasker(model, self, threshold=threshold, tf_idf_map=mapper)
        #更新 self.masker 属性，将其替换为一个新的 TFIDFMasker 对象

    def export_model(self, pruned_model_path, pruned_mask_path):
        pruned_model_state_dict = self.model.state_dict()
        pruned_mask = self.masker.get_tf_idf_mask()

        # 保存剪枝后的模型参数
        torch.save(pruned_model_state_dict, pruned_model_path)

        # 保存剪枝后的掩码
        with open(pruned_mask_path, 'wb') as f:
            torch.save(pruned_mask, f)

        print("Pruned model and mask exported successfully.")


class TFIDFMasker(L1NormPruner):
    def __init__(self, model, pruner, threshold, tf_idf_map, config_list, preserve_round=1, dependency_aware=False ):
        super().__init__(model, config_list)
        # self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
        #                  existed_wrappers: Dict[str, ModuleWrapper] | None = None
        self.threshold = threshold
        self.pruner = pruner
        self.tf_idf_map = tf_idf_map
        self.preserve_round = preserve_round
        self.dependency_aware = dependency_aware

    # 用于生成剪枝掩码，接收参数base_mask（基础掩码）、weight（权重）、
    # num_prune（剪枝数量）、wrapper（包装器）、wrapper_idx（包装器索引）等。
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter -> importance for each filter
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)

        # 根据TF-IDF掩码和阈值生成布尔型掩码
        mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
                      :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

    #从TF-IDF映射中获取指定包装器的TF-IDF掩码
    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        if wrapper.name.split('.')[-1] == 'module':
            name = wrapper.name[0:-7]
        # print(name)
        w_tf_idf_structured = self.tf_idf_map[name]
        return w_tf_idf_structured
