from __future__ import annotations

import torch.nn as nn


# ---------------------------------------------------------
# LSTM baseline 모델
# ---------------------------------------------------------
# window 단위 센서 시계열을 입력받아 마지막 hidden state로 행동 클래스를 분류합니다.
class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, window_size, num_features)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
