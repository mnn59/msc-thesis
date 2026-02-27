"""
Distribution layers - EXACTLY same as original DRL-OR.
DRL-OR's partially-unique output layers preserved without modification.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    def forward(self, x):
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)

class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node):
        super(MultiCategorical, self).__init__()
        linears = []
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        for i in range(num_node):
            linears.append(init_(nn.Linear(num_inputs, num_outputs)))
        self.linears = nn.ModuleList(linears)
    def forward(self, x, dst_state):
        xs = [linear(x) for linear in self.linears]
        concat_x = torch.stack(xs, -2)
        result = concat_x * dst_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(logits=result)

class MultiTypeCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node, num_type):
        super(MultiTypeCategorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        dst_type_linears = []
        for i in range(num_node * num_type):
            dst_type_linears.append(init_(nn.Linear(num_inputs, num_outputs)))
        self.dst_type_linears = nn.ModuleList(dst_type_linears)
    def forward(self, x, dst_state, type_state):
        xs = [linear(x) for linear in self.dst_type_linears]
        concat_x = torch.stack(xs, -2)
        dst_type_state = torch.matmul(dst_state.unsqueeze(-1), type_state.unsqueeze(-2))
        ss = list(dst_type_state.size())
        if len(ss) == 4:
            dst_type_state = dst_type_state.view(ss[0], ss[1], -1)
        elif len(ss) == 3:
            dst_type_state = dst_type_state.view(ss[0], -1)
        else:
            raise NotImplementedError
        result = concat_x * dst_type_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(logits=result)