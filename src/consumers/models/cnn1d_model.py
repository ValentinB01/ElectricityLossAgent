from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128), kernels=(7, 5, 3)):
        super().__init__()
        blocks = []
        previous = 2
        for i, (width, kernel) in enumerate(zip(channels, kernels)):
            blocks += [nn.Conv1d(previous, width, kernel, padding=kernel // 2), nn.BatchNorm1d(width), nn.ReLU()]
            if i < 2:
                blocks.append(nn.MaxPool1d(2))
            previous = width
        blocks += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.layers = nn.Sequential(*blocks)
        self.output_dim = channels[-1]

    def forward(self, sequence):
        return self.layers(sequence)


class CNN1DClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = CNNEncoder(config['channels'], config['kernels'])
        self.classifier = nn.Sequential(nn.Linear(self.encoder.output_dim, 64), nn.ReLU(),
                                        nn.Dropout(config['dropout']), nn.Linear(64, 1))

    def forward(self, sequence, features=None):
        return self.classifier(self.encoder(sequence)).squeeze(-1)
