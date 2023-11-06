import torch.nn as nn
import torch
import numpy as np


class MLP(nn.Module):
    def __init__(self, shape, num_layers=3, num_units=16):
        super().__init__()

        c, h, w = shape
        self.shape = shape

        layers = [nn.Flatten(),
                  nn.Linear(c * h * w, num_units),
                  nn.Softplus(), ]

        for i in range(num_layers - 2):
            layers.append(nn.Linear(num_units, num_units))
            layers.append(nn.Softplus())

        layers.append(nn.Linear(num_units, c * h * w))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1, self.shape[0], self.shape[1], self.shape[2])


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, encoding_functions=6, sigmas=(2.0, ), include_input=True):
        super().__init__()
        bands = torch.linspace(0.0, encoding_functions - 1, encoding_functions) / encoding_functions
        bands = bands.unsqueeze(0)
        self.sigmas = sigmas
        self.register_buffer('bands', bands)
        self.include_input = include_input
        self.dim = encoding_functions * 2 * input_dim * len(sigmas) + input_dim * include_input

    def forward(self, x):
        encoding = []
        if self.include_input:
            encoding.append(x)

        for sigma in self.sigmas:
            value = 2.0 * np.pi * x.unsqueeze(-1) * sigma**self.bands
            value = value.reshape(x.shape[0], -1)
            for func in (torch.sin, torch.cos):
                encoding.append(func(value))
        return torch.cat(encoding, -1)