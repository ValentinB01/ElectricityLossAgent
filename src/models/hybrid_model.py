from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .cnn1d_model import CNNEncoder


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and len(x) == 1:
            # Keep the final consumer instead of dropping a singleton training batch.
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                training=False, momentum=self.momentum, eps=self.eps)
        return super().forward(x)


class HybridClassifier(nn.Module):
    def __init__(self, feature_count: int, config: dict):
        super().__init__()
        self.encoder = CNNEncoder(config['channels'], config['kernels'])
        self.statistics = nn.Sequential(nn.Linear(feature_count, 128), SafeBatchNorm1d(128), nn.ReLU(),
                                        nn.Dropout(.2), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(self.encoder.output_dim + 64, 64), nn.ReLU(),
                                        nn.Dropout(config['dropout']), nn.Linear(64, 1))

    def forward(self, sequence, features):
        return self.classifier(torch.cat([self.encoder(sequence), self.statistics(features)], dim=1)).squeeze(-1)
