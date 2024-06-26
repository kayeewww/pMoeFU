# import torch
# from torch.utils_dic.data import DataLoader, Dataset
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils_dic.data import Dataset,TensorDataset
# from torchvision import datasets, transforms
# from torchtext.data.utils_dic import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.datasets import AG_NEWS
# from torchdata.datapipes.iter import IterableWrapper
# from torch.utils_dic.data import DataLoader
# import re
# # import tensorflow_federated as tff
#
# import numpy as np
# import pandas as pd
#
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
#
# from data_preprocess import data_set
#
# train_data, test_data = data_set("shakespeare")
# def get_data(data):
#     for client_id in data.client_ids:
#         for example in data.create_tf_dataset_for_client(client_id):
#             yield example
# # 定义分词器和字段
# tokenizer = get_tokenizer("basic_english")
# # 构建词汇表
# def yield_tokens(data_iter):
#     for text in data_iter:
#         yield tokenizer(text['snippets'].numpy().decode('utf-8'))
# vocab = build_vocab_from_iterator(yield_tokens(get_data(train_data)), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])
#
# class ShakespeareDataset(Dataset):
#     def __init__(self, data):
#         self.data = list(get_data(data))
#         self.tokenizer = get_tokenizer("basic_english")
#         self.vocab = vocab
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text = self.data[idx]['snippets'].numpy().decode('utf-8')
#         tokens = torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.long)
#         label = torch.tensor(0)  # 标签可以根据你的任务定义
#         return tokens, label
# # 数据处理函数
# def process_data(data):
#     return [(torch.tensor(vocab(tokenizer(text)), dtype=torch.long), torch.tensor(label, dtype=torch.long)) for
#             label, text in data]
#
# def set_para_vocab():
#     vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
#     vocab.set_default_index(vocab["<unk>"])
#     vocab_size = len(vocab)
#     embed_size = 128
#     hidden_size = 256
#     num_layers = 2
#     num_classes = 1  # 对于二分类
#     return vocab_size, embed_size, hidden_size, num_layers, num_classes
