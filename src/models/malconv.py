import torch
from torch import nn


class MalConv(nn.Module):
    def __init__(
        self,
        input_length: int = 2**20,
        window_size: int = 500,
        stride: int = 500,
        channels: int = 128,
        embed_size: int = 8,
        output_size: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.embed = nn.Embedding(257, embed_size, padding_idx=0)
        in_channels = int(embed_size / 2)
        self.conv_1 = nn.Conv1d(
            in_channels, channels, window_size, stride=stride, bias=True
        )
        self.conv_2 = nn.Conv1d(
            in_channels, channels, window_size, stride=stride, bias=True
        )
        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, output_size)
        self.sigmoid = nn.Sigmoid()
        # num_classes
        self.num_classes = output_size

    def forward(self, x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, self.channels)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

    def features(self, x):
        """
        Extracts (flattened) features before the last fully connected layer.
        """
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, self.channels)
        x = self.fc_1(x)
        return x
