import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, stride=1, padding='same', bias=True, dropout=False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x

class fc_block(nn.Module):
    def __init__(self, in_features, out_features, dropout=False, activation=nn.ReLU()) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x