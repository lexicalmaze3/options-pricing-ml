import torch
import torch.nn as nn

N_FEATURES = 8
D_MODEL = 64


class OptionsTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(N_FEATURES)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 8)
        tokens = torch.stack(
            [self.embeddings[i](x[:, i:i+1]) for i in range(N_FEATURES)], dim=1
        )  # (batch, 8, d_model)
        out = self.encoder(tokens)          # (batch, 8, d_model)
        pooled = out.mean(dim=1)            # (batch, d_model)
        return self.head(pooled).squeeze(-1)
