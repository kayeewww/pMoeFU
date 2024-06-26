import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the largest input and output sizes
largest_input_size = (3, 32, 32)
largest_output_size = (64, 32, 32)
flattened_input_size = torch.prod(torch.tensor(largest_input_size)).item()

class MoEModel(nn.Module):
    def __init__(self, experts):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.gating_network = nn.Linear(3 * 32 * 32, self.num_experts)

    def forward(self, x):  # 根据输入的数据/data和dataloader来选择对应的experts，之后再进行门控选择要遗忘的客户端
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Get gating weights
        gates = F.softmax(self.gating_network(x_flat), dim=1)

        # Dispatch inputs to experts and get expert outputs
        dispatcher = SparseDispatcher(self.num_experts, gates, x_flat.size(1))
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [expert(expert_inputs[i]) for i, expert in enumerate(self.experts)]

        # Combine expert outputs
        outputs = dispatcher.combine(expert_outputs)
        return outputs

class SparseDispatcher:
    def __init__(self, num_experts, gates, flattened_input_size):
        self.num_experts = num_experts
        self.gates = gates
        self.batch_size = gates.size(0)
        self.flattened_input_size = flattened_input_size

    def dispatch(self, x):
        # Replicate the input for each expert
        expert_inputs = [x for _ in range(self.num_experts)]
        return expert_inputs

    def expert_to_gates(self):
        return self.gates

    def combine(self, expert_outputs):
        # Combine the expert outputs weighted by the gates
        combined = sum([g.unsqueeze(1).unsqueeze(2).unsqueeze(3) * o for g, o in zip(self.gates.t(), expert_outputs)])
        return combined

# Example expert models with different input and output sizes
class Expert1(nn.Module):
    def __init__(self):
        super(Expert1, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * 16 * 16, 64 * 32 * 32)

    def forward(self, x):
        print(f"Expert1 input shape: {x.shape}")
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 64 * 16 * 16)
        print(f"Expert1 reshaped input shape: {x.shape}")
        x = self.fc(x)
        return x.view(-1, 64, 32, 32)

class Expert2(nn.Module):
    def __init__(self):
        super(Expert2, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * 16 * 16, 64 * 32 * 32)

    def forward(self, x):
        print(f"Expert2 input shape: {x.shape}")
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 32 * 16 * 16)
        print(f"Expert2 reshaped input shape: {x.shape}")
        x = self.fc(x)
        return x.view(-1, 64, 32, 32)


class Net_cifar10(nn.Module):

    def __init__(self):
        super(Net_cifar10, self).__init__()
        # self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64 * 32 * 32)#, 10)
        # self.flatten = nn.Flatten()
        # self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Expert1 input shape: {x.shape}")
        # x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.flatten(x)
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 16 * 5 * 5)
        print(f"Expert1 reshaped input shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 64, 32, 32)


class Net_mnist(nn.Module):
    def __init__(self, num_classes=10):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        # self.fc2 = nn.Linear(128)
        self.fc2 = nn.Linear(128, 64 * 32 * 32)# num_classes)
        # self.input_size = (1, 28, 28)
        # self.output_size = num_classes

    def forward(self, x):
        print(f"Expert2 input shape: {x.shape}")
        x = x[:, 0, :, :].unsqueeze(1)
        print(f"Expert2 input One channel: {x.shape}")

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        print(f"Expert2 reshaped input shape: {x.shape}")  # 64*32*32
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x.view(-1, 64, 32, 32)
        # x = x.view(x.size(0), 1, 10, 10)  # Adjusting the output to have single channel
        x = x.repeat(16, 64, 1, 1)  # Repeating channels to make it 3-channel output
        x = F.interpolate(x, size=(32, 32))  # Resize to (3, 32, 32)
        return x.view(-1, 64, 32, 32)
class NoisyTopKGating:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def __call__(self, x, training):
        # Simplified gating mechanism for illustration purposes
        batch_size = x.size(0)
        gates = torch.softmax(torch.randn(batch_size, self.num_experts), dim=1)
        load = gates.sum(0)
        return gates, load

# Instantiate experts and MoE model

# experts = [Expert1(), Expert2()]
experts = [Net_mnist(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10(), Net_cifar10()]
noisy_top_k_gating = NoisyTopKGating(num_experts=10)
moe_model = MoEModel(experts=experts)

# Example input data
input_data = torch.randn(16, 3, 32, 32)  # Example input for all experts

# Forward pass for input data
output, loss = moe_model(input_data.view(16, -1))
print(output.shape)
print(loss)
