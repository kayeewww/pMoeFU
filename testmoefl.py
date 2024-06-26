import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 定义CIFAR-10专家模型
class CIFAR10Expert(nn.Module):
    def __init__(self):
        super(CIFAR10Expert, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义MNIST专家模型
class MNISTExpert(nn.Module):
    def __init__(self):
        super(MNISTExpert, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载CIFAR-10和MNIST数据集
def load_data():
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    trainloader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=100, shuffle=True, num_workers=2)
    trainset_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=100, shuffle=True, num_workers=2)
    return trainloader_cifar10, trainloader_mnist

# 定义训练函数
def train_expert(expert, trainloader, optimizer, criterion, num_epochs):
    expert.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = expert(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')


# 定义组合模型
class CombinedModel(nn.Module):
    def __init__(self, cifar10_expert, mnist_expert):
        super(CombinedModel, self).__init__()
        self.cifar10_expert = cifar10_expert
        self.mnist_expert = mnist_expert

    def forward(self, x):
        if x.size(1) == 3:  # CIFAR-10 数据
            return self.cifar10_expert(x)
        elif x.size(1) == 1:  # MNIST 数据
            return self.mnist_expert(x)
        else:
            raise ValueError("Unsupported input channel size: {}".format(x.size(1)))

def aggregate_model_parameters(params_list):
    aggregated_params = params_list[0]
    for key in aggregated_params.keys():
        for i in range(1, len(params_list)):
            aggregated_params[key] += params_list[i][key]
        aggregated_params[key] = torch.div(aggregated_params[key], len(params_list))
    return aggregated_params

def calculate_tfidf_scores(model, server_data):
    tfidf_scores = []
    with torch.no_grad():
        for data in server_data:
            output = model(data)
            score = np.random.rand()  # 示例：使用随机数作为TF-IDF分数
            tfidf_scores.append(score)
    return tfidf_scores

def server_inference(model, server_data):
    model.eval()
    with torch.no_grad():
        outputs = model(server_data)
    return outputs

# 上传模型参数到服务器
def upload_model_parameters(model):
    cifar10_params = model.cifar10_expert.state_dict()
    mnist_params = model.mnist_expert.state_dict()
    return cifar10_params, mnist_params

if __name__ == "__main__":
    # 实例化专家模型并进行训练
    trainloader_cifar10, trainloader_mnist = load_data()
    cifar10_expert = CIFAR10Expert()
    mnist_expert = MNISTExpert()

    # 训练CIFAR-10专家模型
    optimizer_cifar10 = optim.SGD(cifar10_expert.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print("Training CIFAR-10 Expert")
    train_expert(cifar10_expert, trainloader_cifar10, optimizer_cifar10, criterion, num_epochs=10)

    # 训练MNIST专家模型
    optimizer_mnist = optim.SGD(mnist_expert.parameters(), lr=0.01, momentum=0.9)
    print("Training MNIST Expert")
    train_expert(mnist_expert, trainloader_mnist, optimizer_mnist, criterion, num_epochs=10)
    # 实例化组合模型
    combined_model = CombinedModel(cifar10_expert, mnist_expert)
    cifar10_params, mnist_params = upload_model_parameters(combined_model)

    # 聚合客户端模型参数
    cifar10_params_list = [cifar10_params]
    mnist_params_list = [mnist_params]
    aggregated_cifar10_params = aggregate_model_parameters(cifar10_params_list)
    aggregated_mnist_params = aggregate_model_parameters(mnist_params_list)

    # 实例化聚合后的专家模型
    cifar10_expert = CIFAR10Expert()
    mnist_expert = MNISTExpert()
    cifar10_expert.load_state_dict(aggregated_cifar10_params)
    mnist_expert.load_state_dict(aggregated_mnist_params)

    # 实例化组合模型
    combined_model = CombinedModel(cifar10_expert, mnist_expert)

    # 计算TF-IDF分数
    server_data_cifar10 = torch.randn(100, 3, 32, 32)  # 示例：服务器端数据
    server_data_mnist = torch.randn(100, 1, 28, 28)  # 示例：服务器端数据

    tfidf_scores_cifar10 = calculate_tfidf_scores(combined_model, server_data_cifar10)
    tfidf_scores_mnist = calculate_tfidf_scores(combined_model, server_data_mnist)

    # 进行模型推理
    output_cifar10 = server_inference(combined_model, server_data_cifar10)
    output_mnist = server_inference(combined_model, server_data_mnist)

    print("Server inference output for CIFAR-10:", output_cifar10)
    print("Server inference output for MNIST:", output_mnist)
