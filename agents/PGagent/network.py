import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, vector_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.dense1 = nn.Linear(vector_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x
