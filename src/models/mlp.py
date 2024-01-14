from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        output_size: int = 8,
        input_size: int = 2381,
        hidden_units: list[int] = [1024, 512, 256],
    ):
        super().__init__()
        all_layers = []
        for hidden in hidden_units:
            all_layers.append(nn.Linear(input_size, hidden))
            all_layers.append(nn.BatchNorm1d(hidden)),
            all_layers.append(nn.ReLU())
            input_size = hidden
        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        self.model = nn.Sequential(*all_layers)
        # num_classes
        self.num_classes = output_size

    def forward(self, x):
        batch_size, _ = x.size()
        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)

    def features(self, x):
        """
        Extracts (flattened) features before the last fully connected layer.
        """
        batch_size, _ = x.size()
        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        fea = self.model[:-1]
        return fea(x)
