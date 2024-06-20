import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from moe import MoE  # 假设 MoE 是自定义的混合专家模型类
# from my_data_preprocess import data_preprocess  # 假设这些是自定义的数据处理函数
from data_preprocess import Net_cifar10, BabyGPTmodel, GPTConfig, load_data, Net_mnist


# 自定义的模型初始化函数
def em_init(dataset_name, FL_params):
    device = torch.device("cuda：2" if torch.cuda.is_available() else "cpu")
    # dataset_name = FL_params.datasets[-1]
    if dataset_name == 'cifar10':
        model = torchvision.models.resnet18(num_classes=10)
        input_size = (3, 32, 32) #3*32*32
        output_size = 10
    elif dataset_name == 'mnist':
        model = Net_mnist()
        input_size = (1, 28, 28) #1 * 28 * 28
        output_size = 10
    elif dataset_name == 'shakespeare':
        model = ShakespeareModel(vocab_size=65)  # 假设 ShakespeareModel 是为 Shakespeare 数据集定义的模型
        input_size = (128, 1, 1)  # 假设输入的块大小为128
        output_size = 65  # 假设输出的字符数量为65
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model.to(device), input_size, output_size

def emloader_init(dataset_name, FL_params):
    device = torch.device("cuda：2" if torch.cuda.is_available() else "cpu")
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2)
    elif dataset_name == 'shakespeare':
        # 自定义 Shakespeare 数据集和数据加载器
        # 这里假设你有一个自定义的 Dataset 类 ShakespeareDataset
        trainset = ShakespeareDataset('./data/shakespeare.txt', seq_length=100)
        trainloader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainloader


class ShakespeareDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, 'r') as f:
            text = f.read()

        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(ShakespeareModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden