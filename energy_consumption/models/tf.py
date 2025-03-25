import torch
import torch.nn as nn
import math

class TF_ECF(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        forecast_steps: int = 1,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, forecast_steps)

    def forward(self, x):
        x = self.input_proj(x)

        pos_emb = self._get_positional_encoding(x.size(1), x.size(2), x.device)
        x = x + pos_emb

        x = self.encoder(x)
        x = x[:, -1, :]
        out = self.output_layer(x)

        return out if self.forecast_steps > 1 else out.squeeze(-1)

    def _get_positional_encoding(self, seq_len: int, d_model: int, device: torch.device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)