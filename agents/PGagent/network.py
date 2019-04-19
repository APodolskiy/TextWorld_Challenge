import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetwork(nn.Module):
    def __init__(self, vector_size, hidden_size, output_size):
        super(FCNetwork, self).__init__()
        self.dense1 = nn.Linear(vector_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x


class RNNNetwork(nn.Module):
    def __init__(self, vector_size, num_cells, hidden_size):
        super(RNNNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size=vector_size, hidden_size=hidden_size)


