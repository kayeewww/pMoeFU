
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional
import copy
from torch.utils.data import Dataset, DataLoader

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False

batch_size = 64
global_max_iters = 20#300
local_max_iters = 2#500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 2#500
eval_iters = 2#200
dropout = 0.2

torch.manual_seed(1337)

# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return text
def load_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    words = open(file_path, 'r', encoding='utf-8').read()
    chars = sorted(list(set(words)))
    vocab_size = len(chars)
    string2integer = {ch: i for i, ch in enumerate(chars)}
    integer2string = {i: ch for ch, i in string2integer.items()}
    encode = lambda s: [string2integer[c] for c in s]
    data = torch.tensor(encode(words), dtype=torch.long)
    return text,data, string2integer, integer2string, vocab_size, chars

def split_data(data: str, num_clients: int):
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    train_chunk_size = len(train_data) // num_clients
    val_chunk_size = len(val_data) // num_clients

    clients_data = {}

    for i in range(num_clients):
        start_train_idx = i * train_chunk_size
        end_train_idx = start_train_idx + train_chunk_size
        start_val_idx = i * val_chunk_size
        end_val_idx = start_val_idx + val_chunk_size

        clients_data[f'client_{i + 1}'] = {
            'train': train_data[start_train_idx:end_train_idx],
            'val': val_data[start_val_idx:end_val_idx]
        }
    return clients_data, train_data, val_data

def get_client_batch(client, split, config, clients_data):
    data = clients_data[client][split]
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_batch(split, config, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, config, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, config, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.atten = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.atten(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.projection(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class BabyGPTmodel(nn.Module):
    def __init__(self, config):
        super(BabyGPTmodel, self).__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Transformer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-12)
        self.lnum_heads = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        tok_emb = self.token(idx)
        position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.positional_embeddings(position_ids)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lnum_heads(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train_model(chars, clients_data, train_data, val_data):
    config = GPTConfig(
        block_size=4,
        vocab_size=len(chars),
        n_head=4,
        n_layer=4,
        n_embd=16
    )

    global_model = BabyGPTmodel(config)
    global_model.to(device)

    clients_models = {'client_1': copy.deepcopy(global_model), 'client_2': copy.deepcopy(global_model), 'client_3': copy.deepcopy(global_model)}

    optimizer = torch.optim.AdamW(global_model.parameters(), lr=learning_rate)

    for iter in range(global_max_iters):
        local_weights = []
        for client in clients_data.keys():
            local_model = BabyGPTmodel(config)
            local_model.to(device)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=learning_rate)
            for local_iter in range(local_max_iters):
                xb, yb = get_client_batch(client, 'train', config, clients_data)
                logits, loss = local_model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            local_weights.append(local_model.state_dict())
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        if iter % 20 == 0 or iter == global_max_iters - 1:
            losses = estimate_loss(global_model, config, train_data, val_data)
            print(f"Global step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        for client in clients_models.keys():
            clients_models[client] = copy.deepcopy(global_model)

    return global_model

def generate_text(model, decode):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(model.generate(context, max_new_tokens=500)[0].tolist())
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

def get_dataloaders(train_data, val_data, block_size, batch_size):
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
# def do_shake():
    text, data, string2integer, integer2string, vocab_size, chars = load_data("data/shakespeare.txt")
    clients_data, train_data, val_data = split_data(data, num_clients=10)
    global_model = train_model(chars, clients_data, train_data, val_data)

    block_size=4
    train_loader, val_loader=get_dataloaders(train_data, val_data, block_size, batch_size)
    print(type(train_loader),type(val_loader), global_model)
    decode = lambda l: ''.join([integer2string[i] for i in l])
    generated_text = generate_text(global_model, decode)
    print(generated_text)