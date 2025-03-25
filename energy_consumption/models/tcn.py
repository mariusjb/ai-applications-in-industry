import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, padding, dropout
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_ECF(nn.Module):
    def __init__(
        self,
        input_size,
        num_channels=[64] * 3,
        kernel_size=3,
        dropout=0.2,
        forecast_steps=1,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers += [
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_steps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]
        output = self.fc(out)

        if self.forecast_steps == 1:
            return output.squeeze(-1)
        else:
            return output
