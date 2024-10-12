import torch
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools
import time
import os


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class GateSigmoid(nn.Module):
    def __init__(self, dim_in):
        super(GateSigmoid, self).__init__()
        self.layer_input = nn.Linear(dim_in, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.activation(x)
        return x


class MLP2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP2, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class GateMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(GateMLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=args.p)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = nn.LogSoftmax(dim=1)
        self.output_size = 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2


class GateResNet(models.ResNet):
    def __init__(self, args):
        super(GateResNet, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])
        self.fc = nn.Linear(512, 10)  # 修改最后一层的输出为10类（CIFAR-10）
        self.activation = nn.LogSoftmax(dim=1)
        self.tfidf_scores = None

    def update_with_tfidf(self, tfidf_scores):
        """
        更新 gate 模型的 tf-idf 分数
        """
        self.tfidf_scores = tfidf_scores

    def forward(self, x):
        x = super(GateResNet, self).forward(x)
        x = self.activation(x)
        return x


class GateCNN(nn.Module):
    def __init__(self, args):
        super(GateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = nn.LogSoftmax(dim=1)
        self.tfidf_scores = None

    def update_with_tfidf(self, tfidf_scores):
        """
        更新 gate 模型的 tf-idf 分数
        """
        self.tfidf_scores = tfidf_scores
        # print("Updated GateCNN with tf-idf scores:", self.tfidf_scores)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2

class ExpertModel(nn.Module):
    def __init__(self, input_shape, conv_layers, linear_layers):
        super(ExpertModel, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_shape[0]

        # 卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        if conv_layers!=0:
            for out_channels, kernel_size in conv_layers:
                self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                self.conv_layers.append(nn.MaxPool2d(2, 2))
                in_channels = out_channels

        # 计算卷积层输出的特征尺寸
        conv_output_size = self._get_conv_output_size(input_shape, self.conv_layers)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        input_dim = conv_output_size
        for output_dim in linear_layers:
            self.fc_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        # 最终输出层
        self.final_fc = nn.Linear(input_dim, 1)  # 输出权重

    def _get_conv_output_size(self, input_shape, conv_layers):
        # 用于计算卷积层后的特征尺寸
        input_data = torch.rand(1, *input_shape)
        for layer in self.conv_layers:
            input_data = F.relu(layer(input_data)) if isinstance(layer, nn.Conv2d) else layer(input_data)
        return int(torch.prod(torch.tensor(input_data.shape[1:])))

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x)) if isinstance(layer, nn.Conv2d) else layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        x = self.final_fc(x)
        return x


class GateModel(nn.Module):
    def __init__(self, input_shape, conv_layers, linear_layers, tfidf_dim, num_experts, FL_params):
        super(GateModel, self).__init__()
        self.input_shape=input_shape
        self.conv_layers=conv_layers
        self.linear_layers=linear_layers

        self.experts = nn.ModuleList([ExpertModel(input_shape, conv_layers, linear_layers) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.tfidf_dim=tfidf_dim
        self.gate = nn.Linear(self.tfidf_dim, num_experts)
        self.tfidf_scores = []

        self.FL_params=FL_params

    def acculumate_feature(self, model, loader, stop: int):
        device = next(model.parameters()).device
        model.to(device)
        features = {}
        classes = []
        all_features = []
        all_classes = []

        gamma = 2.0
        beta = 1.0

        # Define BN layer with gamma and beta
        bn_layer = nn.BatchNorm2d(num_features=model.fc3.in_features, affine=True).to(
            device)  # Assuming 2D conv features
        # Set the BN gamma (weight) and beta (bias) parameters
        with torch.no_grad():
            bn_layer.weight.fill_(gamma)  # Set gamma
            bn_layer.bias.fill_(beta)  # Set beta

        def hook_func(m, x, y, name, feature_iit):
            f = F.relu(y)
            if f.size()[3] != 0:
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
        if isinstance(loader, dict):
            for user_id, loader in loader.items():
                for (inputs, targets) in loader:
                    inputs = inputs.clone().detach().to(device).float()
                    targets = targets.clone().detach().to(device).long()

                    model.eval()
                    classes.extend(targets.cpu().numpy())
                    with torch.no_grad():
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        all_features.append(outputs)
                        all_classes.append(targets)
        else:
            for batch_idx, (inputs, targets) in enumerate(loader):
                # inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                # targets = torch.tensor(targets, dtype=torch.long).to(device)
                inputs = inputs.clone().detach().to(device).float()
                targets = targets.clone().detach().to(device).long()

                if batch_idx == stop:
                    break
                model.eval()
                classes.extend(targets.cpu().numpy())
                with torch.no_grad():
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    all_features.append(outputs)
                    all_classes.append(targets)

        all_features = torch.cat(all_features, dim=0)
        all_classes = torch.cat(all_classes, dim=0)
        # print(f"Accumulated features shape: {all_features.shape}, classes shape: {all_classes.shape}")

        [k.remove() for k in handler_list]
        # return torch.cat(features), torch.cat(classes)
        # return features, classes
        return all_features, all_classes

    def calculate_tf_idf(self, class_wise_features):
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
        idf = torch.clamp(idf, min=0)

        # Calculate TF-IDF
        tf_idf = term_frequency * idf.unsqueeze(0)

        return tf_idf

    def calculate_cp(self, features, classes, dataset, coe, unlearn_client, device):
        # Ensure classes are in long format for indexing
        # print(classes)
        # classes = classes[0].long()

        # Assuming features is a tensor of shape [num_samples, feature_dim]
        class_num = max(classes)# + 1

        feature_dim = features.shape[1]  # Assuming features are of shape [num_samples, feature_dim]

        class_wise_features = torch.zeros(class_num, feature_dim, device=device)

        for fea in range(feature_dim):
            for cls in range(class_num):
                class_mask = (classes == cls).float().unsqueeze(1)
                class_wise_features[cls, fea] = torch.mean(features[:, fea].unsqueeze(1) * class_mask)
        # print("Class-wise features:", class_wise_features)
        tf_idf_map = self.calculate_tf_idf(class_wise_features)  # Your TF-IDF calculation logic here
        return tf_idf_map
    def update_with_tfidf(self, tfidf_scores):
        """
        更新 gate 模型的 tf-idf 分数
        """
        self.tfidf_scores = tfidf_scores
        # print("Updated GateCNN with tf-idf scores:", self.tfidf_scores)

    def calculate_tfidf_scores(self, features, classes, idxs_users, user_train_dataloaders, dict_users, mix_l, FL_params):

        # print('idxu_users in calculating tfidf', idxs_users)
        tf_idf_scores=[]
        client_tfidf_scores = []
        for _ in range(self.FL_params.all_classes['first']):
            # client_tfidf_scores = []
            features, classes = self.acculumate_feature(mix_l, user_train_dataloaders, stop=10)
            tf_idf_map = self.calculate_cp(features, classes, dataset=FL_params.data_1, coe=1, unlearn_client=0,
                                           device=FL_params.device)
            client_tfidf_scores.append(tf_idf_map.mean().item())  # 取平均值并转换为标量

            avg_tfidf = sum(client_tfidf_scores) / len(client_tfidf_scores)  # 计算每个客户端的平均 TF-IDF 得分
            tf_idf_scores.append(avg_tfidf)
        for _ in range(self.FL_params.all_classes['second']):

            features, classes = self.acculumate_feature(mix_l, user_train_dataloaders, stop=10)
            tf_idf_map = self.calculate_cp(features, classes, dataset=FL_params.data_2, coe=1, unlearn_client=0,
                                           device=FL_params.device)
            client_tfidf_scores.append(tf_idf_map.mean().item())  # 取平均值并转换为标量

            avg_tfidf = sum(client_tfidf_scores) / len(client_tfidf_scores)  # 计算每个客户端的平均 TF-IDF 得分
            tf_idf_scores.append(avg_tfidf)
        # print('tf_idf_scores',client_tfidf_scores)
        # return tf_idf_scores
        return client_tfidf_scores
        # return calculate_tfidf_scores(idxs_users, user_train_dataloaders, dict_users, mix_l, FL_params)

    def generate_new_instance(self, output):
        """
        基于当前的 output 生成一个新的 GateModel 实例
        """
        # self.tfidf_dim=self.FL_params.all_classes['second']+self.FL_params.all_classes['first']
        new_instance = GateModel(self.input_shape, self.conv_layers, self.linear_layers, self.tfidf_dim,
                                 self.num_experts, self.FL_params)
        new_instance.tfidf_scores = self.tfidf_scores
        return new_instance
    def forward(self, x, idxs_users, user_train_dataloaders, dict_users, mix_l, FL_params):
        features, classes = self.acculumate_feature(mix_l, user_train_dataloaders, stop=10)
        tfidf_scores = self.calculate_tfidf_scores(features, classes, idxs_users, user_train_dataloaders, dict_users,
                                                   mix_l, FL_params)
        self.tfidf_scores = tfidf_scores
        score_tensor = torch.tensor(tfidf_scores, dtype=torch.float32)  # 转换为张量并增加一个维度，使其形状为 (batch_size, 1)
        # 检查张量的形状并根据维度调整
        # print(f"Original score_tensor shape: {score_tensor.shape}")

        # 如果维度不够，需要进行unsqueeze操作
        if len(score_tensor.shape) == 1:
            score_tensor = score_tensor.unsqueeze(0)

        # 检查能否被20整除，然后进行重塑
        if score_tensor.numel() % 20 == 0:
            score_tensor = score_tensor.view(-1, 20)
        else:
            # 计算需要填充的数量
            num_elements = score_tensor.numel()
            padding_size = (20 - num_elements % 20) % 20  # 计算补足到20的数量

            # 如果需要填充，则在张量的末尾添加0
            if padding_size > 0:
                padding = torch.zeros(padding_size, dtype=score_tensor.dtype, device=score_tensor.device)
                score_tensor = torch.cat([score_tensor.view(-1), padding])
            score_tensor = score_tensor.view(-1, 20)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.view(x.size(0), self.num_experts, -1)

        if self.tfidf_scores is not None:
            gate_scores = self.gate(score_tensor)  # (tfidf_dim) -> (num_experts) 20->10
            gate_scores = F.softmax(gate_scores, dim=-1)  # (num_experts)
            gate_scores = gate_scores.unsqueeze(-1)
            output = torch.sum(expert_outputs * gate_scores, dim=1)  # (batch_size, output_dim)
            # fused_output = torch.sum(expert_outputs * gate_scores, dim=1)
            # print("Output shape:", fused_output.shape)

        else:
            output = torch.mean(expert_outputs, dim=1)  # 如果没有TF-IDF分数，取专家平均值
        gate_weight = torch.sigmoid(output)  # 输出权重在0和1之间
        gate_weight = gate_weight.view(x.size(0), -1)

        return gate_scores, output #gate_scores
    # def forward(self, x, idxs_users, user_train_dataloaders, dict_users, mix_l, FL_params):
    #     # 直接对输入数据进行门控机制
    #     expert_outputs = torch.stack([expert(x) for expert in self.experts],
    #                                  dim=1)  # (batch_size, num_experts, output_dim)
    #     expert_outputs = expert_outputs.view(x.size(0), self.num_experts,
    #                                          -1)  # (batch_size, num_experts, flattened_output)
    #
    #     # 通过输入 x 生成门控权重
    #     gate_scores = self.gate(x)  # 输入数据通过门控网络生成门控权重 (batch_size, num_experts)
    #     gate_scores = F.softmax(gate_scores, dim=-1)  # 使用softmax处理，确保权重总和为1 (batch_size, num_experts)
    #
    #     # 使用门控权重对专家输出进行加权求和
    #     gate_scores = gate_scores.unsqueeze(-1)  # 将门控权重调整为 (batch_size, num_experts, 1)
    #     output = torch.sum(expert_outputs * gate_scores, dim=1)  # 对专家输出进行加权求和 (batch_size, output_dim)
    #
    #     gate_weight = torch.sigmoid(output)  # 输出权重在0和1之间
    #     gate_weight = gate_weight.view(x.size(0), -1)
    #
    #     return gate_scores, output  # 返回门控权重和加权求和后的输出


class GateCNNSoftmax(nn.Module):
    def __init__(self, args):
        super(GateCNNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class Net_purchase(nn.Module):
    def __init__(self):
        super(Net_purchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class GateNetPurchase(nn.Module):
    def __init__(self):
        super(GateNetPurchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 1)
        self.activation = nn.Sigmoid()
        self.tfidf_scores = None

    def update_with_tfidf(self, tfidf_scores):
        self.tfidf_scores = tfidf_scores
        # print("Updated GateNetPurchase with tf-idf scores:", self.tfidf_scores)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
class Net_adult(nn.Module):
    def __init__(self):
        super(Net_adult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class GateNetAdult(nn.Module):
    def __init__(self):
        super(GateNetAdult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
        self.activation = nn.Sigmoid()
        self.tfidf_scores = None

    def update_with_tfidf(self, tfidf_scores):
        self.tfidf_scores = tfidf_scores
        # print("Updated GateNetAdult with tf-idf scores:", self.tfidf_scores)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
class CNNFashion(nn.Module):
    def __init__(self, args):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 10)
        self.activation = nn.LogSoftmax(dim=1)
        self.output_size = 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2


class GateCNNFashion(nn.Module):
    def __init__(self, args):
        super(GateCNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 1)
        self.activation = nn.LogSoftmax(dim=1)
        self.tfidf_scores = None

    def update_with_tfidf(self, tfidf_scores):
        """
        更新 gate 模型的 tf-idf 分数
        """
        self.tfidf_scores = tfidf_scores
        # print("Updated GateCNN with tf-idf scores:", self.tfidf_scores)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class GateCNNFahsionSoftmax(nn.Module):
    def __init__(self, args):
        super(GateCNNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 3)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        # self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        # self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 8)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8, 4)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x, offsets):
        x = self.embeddings(x, offsets)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x.view(len(x), 1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.activation(x)
        return out


class LSTMGate(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMGate, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        # self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        # self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 8)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, offsets):
        x = self.embeddings(x, offsets)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x.view(len(x), 1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.activation(x)
        return out


def save_model(net, save_dir, prefix):
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 创建文件名，加入epoch等信息
    filename = f"{prefix}_{timestamp}.pth"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存模型
    torch.save(net.state_dict(), f"{save_dir}/{filename}")

    print(f"Model saved to {save_dir}/{filename}")
