import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.distributions.normal import Normal

# 定义Net_mnist和Net_cifar10模型类
class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SparseDispatcher:
    def __init__(self, num_experts, gates, flattened_input_size):
        self.num_experts = num_experts
        self.gates = gates
        self.batch_size = gates.size(0)
        self.flattened_input_size = flattened_input_size

    def dispatch(self, x):
        total_input_size = x.size(1)
        split_sizes = [total_input_size // self.num_experts] * self.num_experts
        x_splits = torch.split(x, split_sizes, dim=1)
        expert_inputs = []
        for i in range(self.num_experts):
            mask = self.gates[:, i].unsqueeze(1)
            masked_input = x_splits[i] * mask
            expanded_input = self.expand_to_largest_size(masked_input)
            expert_inputs.append(expanded_input)
        return expert_inputs
    # def dispatch(self, x):
    #     expert_inputs = []
    #     for i in range(self.num_experts):
    #         mask = self.gates[:, i].unsqueeze(1)
    #         masked_input = x * mask
    #         expert_inputs.append(masked_input)
    #     return expert_inputs
    def expand_to_largest_size(self, tensor):
        expanded_tensor = torch.zeros(self.batch_size, self.flattened_input_size, device=tensor.device)
        expanded_tensor[:, :tensor.size(1)] = tensor
        return expanded_tensor
    def expert_to_gates(self):
        return self.gates

    def combine(self, expert_outputs):
        combined = torch.stack(expert_outputs, 0).sum(0)
        return combined

class MoE(nn.Module):
    def __init__(self, num_experts, input_size, tf_idf_scores, forget_client_idx, k, noisy_gating=True):
        super(MoE, self).__init__()
        self.forget_client_idx = forget_client_idx
        self.noisy_gating = noisy_gating
        self.tf_idf_scores = tf_idf_scores
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([Net_mnist() for _ in range(num_experts // 2)] + [Net_cifar10() for _ in range(num_experts // 2, num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        device = next(self.parameters()).device
        x = x.to(device)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        for i in range(clean_logits.size(0)):
            client_importance = self.tf_idf_scores[i % len(self.tf_idf_scores)]
            if isinstance(client_importance, dict):
                client_importance = client_importance.get(i, 1.0)
            top_logits[i] *= client_importance

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]

        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        flat_indices = top_indices.flatten()
        unique, counts = flat_indices.unique(return_counts=True)
        least_common_indices = unique[torch.argsort(counts)[:self.k]]
        least_important_indices = torch.argsort(gates.sum(dim=0))[:self.k]
        self.forget_client_idx = least_important_indices.tolist()

        return gates, load

    def forward(self, x, loss_coef=1e-2):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).float()
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        largest_input_size_mnist = (1, 28, 28)
        largest_input_size_cifar10 = (3, 32, 32)
        flattened_input_size_mnist = torch.prod(torch.tensor(largest_input_size_mnist)).item()
        flattened_input_size_cifar10 = torch.prod(torch.tensor(largest_input_size_cifar10)).item()
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, max(flattened_input_size_mnist, flattened_input_size_cifar10))
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if isinstance(self.experts[i], Net_mnist):
                input_size = (1, 28, 28)
                expert_outputs.append(self.experts[i](expert_inputs[i].view(-1, *input_size)))
            elif isinstance(self.experts[i], Net_cifar10):
                input_size = (3, 32, 32)
                expert_outputs.append(self.experts[i](expert_inputs[i].view(-1, *input_size)))

        y = dispatcher.combine(expert_outputs)
        return y, loss

# 定义数据集和数据加载器
transform = transforms.Compose([transforms.ToTensor()])

mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_dataloader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True)

# 初始化MoE模型
num_experts = 4
input_size = 28*28  # MNIST的输入特征数
tf_idf_scores = [1.0] * 64  # 假设所有客户端的TF-IDF分数为1.0
forget_client_idx = []
k = 2

moe_model = MoE(num_experts, input_size, tf_idf_scores, forget_client_idx, k)

# 测试MNIST数据集
for batch in mnist_dataloader:
    images, labels = batch
    outputs, loss = moe_model(images)
    print("Forget Client Indices (MNIST):", moe_model.forget_client_idx)
    break

# CIFAR10 的输入特征数不同，重新初始化MoE模型
input_size = 3*32*32  # CIFAR10的输入特征数
moe_model = MoE(num_experts, input_size, tf_idf_scores, forget_client_idx, k)

# 测试CIFAR10数据集
for batch in cifar10_dataloader:
    images, labels = batch
    outputs, loss = moe_model(images)
    print("Forget Client Indices (CIFAR10):", moe_model.forget_client_idx)
    break
