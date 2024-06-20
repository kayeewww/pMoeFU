# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

import data_preprocess#model_initiation
class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self.num_experts = num_experts
        self.gates = gates
        self.batch_size = gates.size(0)
        self.expert_index = torch.argmax(gates, dim=1)
        self.expert_inputs = [[] for _ in range(num_experts)]
        self.expert_gates = [[] for _ in range(num_experts)]
        for i in range(self.batch_size):
            expert = self.expert_index[i].item()
            self.expert_inputs[expert].append(i)
            self.expert_gates[expert].append(gates[i, expert].item())

    def dispatch(self, x):
        return [x[indices] for indices in self.expert_inputs]

    def expert_to_gates(self):
        return [torch.tensor(gates) for gates in self.expert_gates]

    def combine(self, expert_outputs):
        y = torch.zeros(self.batch_size, expert_outputs[0].size(1), device=expert_outputs[0].device)
        for i, indices in enumerate(self.expert_inputs):
            y[indices] = expert_outputs[i]
        return y
# class MoE(nn.Module):
#     def __init__(self, experts, num_experts, tf_idf_scores, forget_client_idx, k, noisy_gating=True):
#         super(MoE, self).__init__()
#         self.forget_client_idx = forget_client_idx
#         self.noisy_gating = noisy_gating
#         self.tf_idf_scores = tf_idf_scores
#         self.num_experts = num_experts
#         self.k = k
#         self.experts = experts
#         self.w_gate = nn.ParameterList([nn.Parameter(torch.zeros(expert.input_size[0] * expert.input_size[1] * expert.input_size[2], num_experts), requires_grad=True) for expert in experts])
#         self.w_noise = nn.ParameterList([nn.Parameter(torch.zeros(expert.input_size[0] * expert.input_size[1] * expert.input_size[2], num_experts), requires_grad=True) for expert in experts])
#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(dim=1)
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))
#         assert(self.k <= self.num_experts)
#
#     def cv_squared(self, x):
#         eps = 1e-10
#         if x.shape[0] == 1:
#             return torch.tensor([0], device=x.device, dtype=x.dtype)
#         return x.float().var() / (x.float().mean()**2 + eps)
#
#     def _gates_to_load(self, gates):
#         return (gates > 0).sum(0)
#
#     def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
#         batch = clean_values.size(0)
#         m = noisy_top_values.size(1)
#         top_values_flat = noisy_top_values.flatten()
#         threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
#         threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
#         is_in = torch.gt(noisy_values, threshold_if_in)
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
#         normal = Normal(self.mean, self.std)
#         prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
#         prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
#         prob = torch.where(is_in, prob_if_in, prob_if_out)
#         return prob
#
#     def noisy_top_k_gating(self, x, train, expert_idx, noise_epsilon=1e-2):
#         device = next(self.parameters()).device
#         x = x.to(device)
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         clean_logits = x @ self.w_gate[expert_idx]
#         if self.noisy_gating and train:
#             raw_noise_stddev = x @ self.w_noise[expert_idx]
#             noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
#             noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
#             logits = noisy_logits
#         else:
#             logits = clean_logits
#         logits = self.softmax(logits)
#         top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
#         for i in range(clean_logits.size(0)):
#             client_importance = self.tf_idf_scores[i % len(self.tf_idf_scores)]
#             if isinstance(client_importance, dict):
#                 client_importance = client_importance.get(i, 1.0)
#             top_logits[i] *= client_importance
#         top_k_logits = top_logits[:, :self.k]
#         top_k_indices = top_indices[:, :self.k]
#         top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
#         zeros = torch.zeros_like(logits, requires_grad=True)
#         gates = zeros.scatter(1, top_k_indices, top_k_gates)
#         if self.noisy_gating and self.k < self.num_experts and train:
#             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
#         else:
#             load = self._gates_to_load(gates)
#         flat_indices = top_indices.flatten()
#         unique, counts = flat_indices.unique(return_counts=True)
#         least_common_indices = unique[torch.argsort(counts)[:self.k]]
#         self.forget_client_idx = least_common_indices.tolist()
#         return gates, load
#
#     def forward(self, x, expert_idx, loss_coef=1e-2):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1).float()
#         gates, load = self.noisy_top_k_gating(x, self.training, expert_idx)
#         importance = gates.sum(0)
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= loss_coef
#         dispatcher = SparseDispatcher(self.num_experts, gates)
#         expert_inputs = dispatcher.dispatch(x)
#         gates = dispatcher.expert_to_gates()
#         expert_outputs = [self.experts[i](self.reshape_expert_input(expert_inputs[i], self.experts[i].input_size)) for i in range(self.num_experts)]
#         y = dispatcher.combine(expert_outputs)
#         return y, loss
#
#     def reshape_expert_input(self, expert_input, input_size):
#         if len(input_size) == 3:
#             return expert_input.view(-1, *input_size)
#         else:
#             return expert_input

# class SparseDispatcher(object):
#     """Helper for implementing a mixture of experts.
#     The purpose of this class is to create input mini-batches for the
#     experts and to combine the results of the experts to form a unified
#     output tensor.
#     There are two functions:
#     dispatch - take an input Tensor and create input Tensors for each expert.
#     combine - take output Tensors from each expert and form a combined output
#       Tensor.  Outputs from different experts for the same batch element are
#       summed together, weighted by the provided "gates".
#     The class is initialized with a "gates" Tensor, which specifies which
#     batch elements go to which experts, and the weights to use when combining
#     the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
#     The inputs and outputs are all two-dimensional [batch, depth].
#     Caller is responsible for collapsing additional dimensions prior to
#     calling this class and reshaping the output to the original shape.
#     See common_layers.reshape_like().
#     Example use:
#     gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
#     inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
#     experts: a list of length `num_experts` containing sub-networks.
#     dispatcher = SparseDispatcher(num_experts, gates)
#     expert_inputs = dispatcher.dispatch(inputs)
#     expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
#     outputs = dispatcher.combine(expert_outputs)
#     The preceding code sets the output for a particular example b to:
#     output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
#     This class takes advantage of sparsity in the gate matrix by including in the
#     `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
#     """
#
#     def __init__(self, num_experts, gates):
#         """Create a SparseDispatcher."""
#
#         self._gates = gates
#         self._num_experts = num_experts
#         # sort experts
#         sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#         # drop indices
#         _, self._expert_index = sorted_experts.split(1, dim=1)
#         # get according batch index for each expert
#         self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
#         # calculate num samples that each expert gets
#         self._part_sizes = (gates > 0).sum(0).tolist()
#         # expand gates to match with self._batch_index
#         gates_exp = gates[self._batch_index.flatten()]
#         self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
#
#     def dispatch(self, inp):
#         """Create one input Tensor for each expert.
#         The `Tensor` for a expert `i` contains the slices of `inp` corresponding
#         to the batch elements `b` where `gates[b, i] > 0`.
#         Args:
#           inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
#         Returns:
#           a list of `num_experts` `Tensor`s with shapes
#             `[expert_batch_size_i, <extra_input_dims>]`.
#         """
#
#         # assigns samples to experts whose gate is nonzero
#         device = inp.device  # 获取输入张量的设备
#         self._batch_index = self._batch_index.to(device)  # 将索引转换到相同的设备
#
#         # expand according to batch index so we can just split by _part_sizes
#         inp_exp = inp[self._batch_index].squeeze(1)
#         return torch.split(inp_exp, self._part_sizes, dim=0)
#
#     def combine(self, expert_out, multiply_by_gates=True):
#         """Sum together the expert output, weighted by the gates.
#         The slice corresponding to a particular batch element `b` is computed
#         as the sum over all experts `i` of the expert output, weighted by the
#         corresponding gate values.  If `multiply_by_gates` is set to False, the
#         gate values are ignored.
#         Args:
#           expert_out: a list of `num_experts` `Tensor`s, each with shape
#             `[expert_batch_size_i, <extra_output_dims>]`.
#           multiply_by_gates: a boolean
#         Returns:
#           a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
#         """
#         # apply exp to expert outputs, so we are not longer in log space
#         device = expert_out[0].device
#         self._batch_index = self._batch_index.to(device)
#         stitched = torch.cat(expert_out, 0)
#
#         if multiply_by_gates:
#             stitched = stitched.mul(self._nonzero_gates)
#         zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
#         # combine samples that have been processed by the same k experts
#         combined = zeros.index_add(0, self._batch_index, stitched.float())
#         return combined
#
#     def expert_to_gates(self):
#         """Gate values corresponding to the examples in the per-expert `Tensor`s.
#         Returns:
#           a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
#               and shapes `[expert_batch_size_i]`
#         """
#         # split nonzero gates for each expert
#         device = self._nonzero_gates.device
#         self._nonzero_gates = self._nonzero_gates.to(device)
#         return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax(1)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.soft(out)
#         return out


# class MoE(nn.Module):
#
#     """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
#     Args:
#     input_size: integer - size of the input
#     output_size: integer - size of the input
#     num_experts: an integer - number of experts
#     hidden_size: an integer - hidden size of the experts
#     noisy_gating: a boolean
#     k: an integer - how many experts to use for each batch element
#     """
#
#     def __init__(self, experts, num_experts, tf_idf_scores, forget_client_idx, k,
#                  noisy_gating=True):
#     # def __init__(self, input_size, output_size, experts, num_experts, hidden_size, tf_idf_scores, forget_client_idx, k, noisy_gating=True):
#         super(MoE, self).__init__()
#         self.forget_client_idx = forget_client_idx
#         self.noisy_gating = noisy_gating
#         self.tf_idf_scores = tf_idf_scores
#         self.num_experts = num_experts
#         # self.output_size = output_size
#         # self.input_size = input_size
#
#         # self.hidden_size = hidden_size
#         self.k = k
#
#         # instantiate experts
#         self.experts = experts
#         # self.experts = nn.ModuleList([MLP(
#         # self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
#         # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
#         # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
#         # self.w_gate = nn.Parameter(torch.zeros(400, num_experts), requires_grad=True)
#         # self.w_noise = nn.Parameter(torch.zeros(400, num_experts), requires_grad=True)
#         self.w_gate = nn.ParameterList(
#             [nn.Parameter(torch.zeros(expert.input_size, num_experts), requires_grad=True) for expert in experts])
#         self.w_noise = nn.ParameterList(
#             [nn.Parameter(torch.zeros(expert.input_size, num_experts), requires_grad=True) for expert in experts])
#
#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(1)
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))
#         assert(self.k <= self.num_experts)
#
#     def cv_squared(self, x):
#         """The squared coefficient of variation of a sample.
#         Useful as a loss to encourage a positive distribution to be more uniform.
#         Epsilons added for numerical stability.
#         Returns 0 for an empty Tensor.
#         Args:
#         x: a `Tensor`.
#         Returns:
#         a `Scalar`.
#         """
#         eps = 1e-10
#         # if only num_experts = 1
#
#         if x.shape[0] == 1:
#             return torch.tensor([0], device=x.device, dtype=x.dtype)
#         return x.float().var() / (x.float().mean()**2 + eps)
#
#     def _gates_to_load(self, gates):
#         """Compute the true load per expert, given the gates.
#         The load is the number of examples for which the corresponding gate is >0.
#         Args:
#         gates: a `Tensor` of shape [batch_size, n]
#         Returns:
#         a float32 `Tensor` of shape [n]
#         """
#         return (gates > 0).sum(0)
#
#     def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
#         """Helper function to NoisyTopKGating.
#         Computes the probability that value is in top k, given different random noise.
#         This gives us a way of backpropagating from a loss that balances the number
#         of times each expert is in the top k experts per example.
#         In the case of no noise, pass in None for noise_stddev, and the result will
#         not be differentiable.
#         Args:
#         clean_values: a `Tensor` of shape [batch, n].
#         noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
#           normally distributed noise with standard deviation noise_stddev.
#         noise_stddev: a `Tensor` of shape [batch, n], or None
#         noisy_top_values: a `Tensor` of shape [batch, m].
#            "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
#         Returns:
#         a `Tensor` of shape [batch, n].
#         """
#         batch = clean_values.size(0)
#         m = noisy_top_values.size(1)
#         top_values_flat = noisy_top_values.flatten()
#
#         threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
#         threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
#         is_in = torch.gt(noisy_values, threshold_if_in)
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
#         # is each value currently in the top k.
#         normal = Normal(self.mean, self.std)
#         prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
#         prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
#         if torch.isnan(prob_if_in).any() or torch.isnan(prob_if_out).any():
#             print("NaN detected in prob_if_in or prob_if_out")
#             print(f"clean_values: {clean_values}")
#             print(f"threshold_if_in: {threshold_if_in}")
#             print(f"threshold_if_out: {threshold_if_out}")
#             print(f"noise_stddev: {noise_stddev}")
#         prob = torch.where(is_in, prob_if_in, prob_if_out)
#         return prob
#
#     def noisy_top_k_gating(self, x, train, expert_idx, noise_epsilon=1e-2):
#         """Noisy top-k gating.
#           See paper: https://arxiv.org/abs/1701.06538.
#           Args:
#             x: input Tensor with shape [batch_size, input_size]
#             train: a boolean - we only add noise at training time.
#             noise_epsilon: a float
#           Returns:
#             gates: a Tensor with shape [batch_size, num_experts]
#             load: a Tensor with shape [num_experts]
#         """
#         device = next(self.parameters()).device
#         x = x.to(device)
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         clean_logits = x @ self.w_gate[expert_idx]
#         # device = next(self.parameters()).device  # 获取模型所在的设备
#         # x = x.to(device)  # 确保输入张量在同一设备上
#         # batch_size = x.size(0)
#         # x = x.view(batch_size, -1)
#         # clean_logits = x @ self.w_gate
#
#         if self.noisy_gating and train:
#             raw_noise_stddev = x @ self.w_noise[expert_idx] #原始噪声标准差
#             noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon)) #噪声标准差
#             noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev) #噪声logits
#             logits = noisy_logits
#         else:
#             logits = clean_logits
#
#         # calculate topk + 1 that will be needed for the noisy gates
#         logits = self.softmax(logits) #对logits应用softmax函数。
#         top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
#         #TF-IDF的部分
#         for i in range(clean_logits.size(0)):
#             client_importance = self.tf_idf_scores[i % len(self.tf_idf_scores)]
#             if isinstance(client_importance, dict):
#                 client_importance = client_importance.get(i, 1.0)  # 提取字典中的数值，如果没有就默认1.0
#                 # Get client importance score from dictionary or default to 1.0
#             top_logits[i] *= client_importance # Multiply top_logits[i] by client importance score
#
#         top_k_logits = top_logits[:, :self.k]
#         top_k_indices = top_indices[:, :self.k]
#
#         # self.forget_client_idx = most_uncommon_index.item()
#         top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
#
#         zeros = torch.zeros_like(logits, requires_grad=True)
#         gates = zeros.scatter(1, top_k_indices, top_k_gates)
#
#         if self.noisy_gating and self.k < self.num_experts and train:
#             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
#         else:
#             load = self._gates_to_load(gates)
#         # Incorporate TF-IDF scores into the selection
#         flat_indices = top_indices.flatten()
#         unique, counts = flat_indices.unique(return_counts=True)
#         least_common_indices = unique[torch.argsort(counts)[:self.k]]
#
#         self.forget_client_idx = least_common_indices.tolist()
#         # print('moe forget check:', least_common_indices)
#
#         return gates, load
#
#     def forward(self, x, expert_idx, loss_coef=1e-2):
#         """Args:
#         x: tensor shape [batch_size, input_size]
#         train: a boolean scalar.
#         loss_coef: a scalar - multiplier on load-balancing losses
#
#         Returns:
#         y: a tensor with shape [batch_size, output_size].
#         extra_training_loss: a scalar.  This should be added into the overall
#         training loss of the model.  The backpropagation of this loss
#         encourages all experts to be approximately equally used across a batch.
#         """
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1).float()
#         gates, load = self.noisy_top_k_gating(x, self.training, expert_idx)
#         # batch_size = x.size(0)
#         # x = x.view(batch_size, -1).float()
#         # gates, load = self.noisy_top_k_gating(x, self.training)
#         # calculate importance loss
#         importance = gates.sum(0)
#
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= loss_coef
#
#         dispatcher = SparseDispatcher(self.num_experts, gates)
#         expert_inputs = dispatcher.dispatch(x)
#         gates = dispatcher.expert_to_gates()
#         # expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
#         # expert_outputs = [self.experts[i](expert_inputs[i].view(-1, 3, 32, 32)) for i in range(self.num_experts)]
#         # expert_outputs = [self.experts[i](expert_inputs[i].view(-1, 1, 28, 28)) for i in range(self.num_experts)]
#         expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
#
#         y = dispatcher.combine(expert_outputs)
#         return y, loss
class MoE(nn.Module):
    def __init__(self, experts, num_experts,input_sizes, output_size, tf_idf_scores, forget_client_idx, k, noisy_gating=True):
        super(MoE, self).__init__()
        self.forget_client_idx = forget_client_idx
        self.noisy_gating = noisy_gating
        self.tf_idf_scores = tf_idf_scores
        self.num_experts = num_experts
        self.k = k
        self.experts = experts
        self.input_sizes = input_sizes
        self.output_size = output_size
        # self.w_gate = nn.ParameterList([nn.Parameter(
        #     torch.zeros(self.input_size[0] * expert.input_size[1] * expert.input_size[2], num_experts),
        #     requires_grad=True) if isinstance(expert.input_size, tuple) else nn.Parameter(
        #     torch.zeros(expert.input_size, num_experts), requires_grad=True) for expert in experts])
        #
        # self.w_noise = nn.ParameterList([nn.Parameter(
        #     torch.zeros(expert.input_size[0] * expert.input_size[1] * expert.input_size[2], num_experts),
        #     requires_grad=True) if isinstance(self.input_size, tuple) else nn.Parameter(
        #     torch.zeros(self.input_size, num_experts), requires_grad=True) for expert in experts])

        # self.w_gate = nn.ParameterList(
        #                 [nn.Parameter(torch.zeros(input_size[input_size], num_experts), requires_grad=True) for input_size in self.input_sizes])
        #
        # self.w_noise = nn.ParameterList(
        #                 [nn.Parameter(torch.zeros(input_size[input_size], num_experts), requires_grad=True) for input_size in self.input_sizes])
        self.w_gate = nn.ParameterList([nn.Parameter(torch.zeros(size[0] * size[1] * size[2], num_experts), requires_grad=True) for size in self.input_sizes])
        self.w_noise = nn.ParameterList([nn.Parameter(torch.zeros(size[0] * size[1] * size[2], num_experts), requires_grad=True) for size in self.input_sizes])
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
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

    def noisy_top_k_gating(self, x, train, expert_idx, noise_epsilon=1e-2):
        device = next(self.parameters()).device
        x = x.to(device)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        clean_logits = x @ self.w_gate[expert_idx]
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise[expert_idx]
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
        self.forget_client_idx = least_common_indices.tolist()
        return gates, load

    # def forward(self, x, expert_idx, loss_coef=1e-2):
    #     batch_size = x.size(0)
    #     input_size = self.input_sizes[expert_idx]
    #     x = x.view(batch_size, -1).float()
    #     gates, load = self.noisy_top_k_gating(x, self.training, expert_idx)
    #     importance = gates.sum(0)
    #     loss = self.cv_squared(importance) + self.cv_squared(load)
    #     loss *= loss_coef
    #     dispatcher = SparseDispatcher(self.num_experts, gates)
    #     expert_inputs = dispatcher.dispatch(x)
    #     gates = dispatcher.expert_to_gates()
    #     view_size=[(-1, 3, 32, 32), (-1, 1, 28, 28), (-1, 128, 1, 1), (-1, 3, 32, 32), (-1, 1, 28, 28), (-1, 128, 1, 1), (-1, 3, 32, 32), (-1, 1, 28, 28), (-1, 128, 1, 1), (-1, 3, 32, 32)]
    #
    #     expert_outputs = [self.experts[i](expert_inputs[i].view(view_size[i])) for i in range(self.num_experts)]
    #     # expert_outputs = [self.experts[i](self.reshape_expert_input(expert_inputs[i], input_size)) for i
    #     #                   in range(self.num_experts)]
    #
    #     y = dispatcher.combine(expert_outputs)
    #     return y, loss
    def forward(self, x, expert_idx, loss_coef=1e-2):
        batch_size = x.size(0)
        input_size0 = self.input_sizes[expert_idx]
        input_size=input_size0[0]*input_size0[1]*input_size0[2]
        x = x.view(batch_size, -1).float()
        # x = self.reshape_expert_input(x, input_size)  # Ensure correct input shape
        gates, load = self.noisy_top_k_gating(x, self.training, expert_idx)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        print(input_size0)
        expert_outputs = [self.experts[i](self.reshape_expert_input(expert_inputs[i], input_size0)) for i in
                          range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

    # def reshape_expert_input(self, expert_input, input_size):
    #     for size in input_size:
    #         if len(size) == 1:
    #             return expert_input.view(-1, *size)
    #         expected_size = torch.prod(torch.tensor(size)).item()
    #         actual_size = expert_input.size(1)
    #         if expected_size != actual_size:
    #             raise RuntimeError(f"Expected input size {expected_size}, but got {actual_size}")
    #         return expert_input.view(-1, *size)

    def reshape_expert_input(self, expert_input, input_size):
        expected_size = torch.prod(torch.tensor(input_size)).item()
        actual_size = expert_input.size(1)
        if expected_size != actual_size:
            raise RuntimeError(f"Expected input size {expected_size}, but got {actual_size}")
        return expert_input.view(-1, *input_size)

