import torch.nn as nn


class LSTM_ECF(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, dropout=0.2, forecast_steps=1
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, forecast_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        output = self.fc(out[:, -1, :])

        if self.forecast_steps == 1:
            return output.squeeze(-1)
        else:
            return output
