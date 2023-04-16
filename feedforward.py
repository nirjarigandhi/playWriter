from __future__ import annotations
import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(FeedForward, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.first = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.last = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.result = None

    def forward(self, input_tensor: torch.Tensor):

        self.result = self.first(input_tensor)
        self.relu = torch.nn.ReLU()
        self.result = self.relu(self.result)
        self.result = self.last(self.result)

        return self.result