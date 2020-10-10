import torch
import torch.nn.functional as F


class DevolutionNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(DevolutionNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.bn1 = torch.nn.BatchNorm1d(num_features=self.input_size)

        self.input = torch.nn.Linear(self.input_size, self.hidden_layer_sizes[0])
        self.hidden = []
        for k in range(len(self.hidden_layer_sizes) - 1):
            self.hidden.append(torch.nn.Linear(self.hidden_layer_sizes[k], self.hidden_layer_sizes[k + 1]))
        self.output = torch.nn.Linear(self.hidden_layer_sizes[-1], self.output_size)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.input(x))

        for layer in self.hidden:
            x = F.relu(layer(x))
        # x = F.softmax(self.output(x), dim=1)
        x = self.output(x)
        return x
