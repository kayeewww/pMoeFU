import torch
import torch.nn as nn
import numpy as np

def aggregate_model_parameters(model_params_list):
    aggregated_params = model_params_list[0]
    for key in aggregated_params.keys():
        for i in range(1, len(model_params_list)):
            aggregated_params[key] += model_params_list[i][key]
        aggregated_params[key] = torch.div(aggregated_params[key], len(model_params_list))
    return aggregated_params

def calculate_tfidf_scores(models, server_data):
    tfidf_scores = []
    for model in models:
        # 使用服务器端数据计算TF-IDF分数
        score = np.random.rand()  # 示例：使用随机数作为TF-IDF分数
        tfidf_scores.append(score)
    return tfidf_scores

def server_inference(models, server_data):
    outputs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(server_data)
            outputs.append(output)
    aggregated_output = torch.stack(outputs, dim=0).mean(dim=0)
    return aggregated_output

# 聚合客户端模型参数
aggregated_params = aggregate_model_parameters([model_params])

# 实例化聚合后的专家模型
aggregated_models = [CIFAR10Expert(), MNISTExpert()]
for model, params in zip(aggregated_models, aggregated_params):
    model.load_state_dict(params)

# 计算TF-IDF分数
server_data = torch.randn(100, 3, 32, 32)  # 示例：服务器端数据
tfidf_scores = calculate_tfidf_scores(aggregated_models, server_data)

# 进行模型推理
output = server_inference(aggregated_models, server_data)
print("Server inference output:", output)
