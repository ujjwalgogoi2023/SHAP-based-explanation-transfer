"""
Hybrid CNN-LSTM intrusion detection classifier.

Architecture: a 1D-CNN feature extractor is applied independently to each
timestep's flow-feature vector (TimeDistributed-style), and the resulting
per-timestep embeddings are fed to an LSTM across the sequence dimension.
With sequence_length=1 (the default — flows treated as independent) this
reduces to CNN feature extraction followed by a single-step LSTM, which is
still a legitimate hybrid CNN-LSTM classifier and remains fully
differentiable for DeepSHAP.

Input:  (batch, seq_len, n_features)
Output: (batch, n_classes) logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        cnn_channels=(32, 64),
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.n_features = n_features

        conv_layers = []
        in_ch = 1
        length = n_features
        for out_ch in cnn_channels:
            pad = cnn_kernel_size // 2
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=pad))
            conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, ceil_mode=True))
            in_ch = out_ch
            length = (length + 1) // 2
        self.cnn = nn.Sequential(*conv_layers)
        self.cnn_out_dim = in_ch * length

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq_len, n_features = x.shape
        x = x.reshape(batch * seq_len, 1, n_features)  # (batch*seq_len, 1, n_features)
        x = self.cnn(x)  # (batch*seq_len, channels, reduced_len)
        x = x.reshape(batch, seq_len, -1)  # (batch, seq_len, cnn_out_dim)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_dim)
        last_step = lstm_out[:, -1, :]  # (batch, lstm_out_dim)

        logits = self.classifier(last_step)  # (batch, n_classes)
        return logits


def build_model(config: dict, n_features: int, n_classes: int) -> CNNLSTM:
    m = config["model"]
    return CNNLSTM(
        n_features=n_features,
        n_classes=n_classes,
        cnn_channels=tuple(m["cnn_channels"]),
        cnn_kernel_size=m["cnn_kernel_size"],
        lstm_hidden_size=m["lstm_hidden_size"],
        lstm_layers=m["lstm_layers"],
        dropout=m["dropout"],
        bidirectional=m["bidirectional"],
    )
