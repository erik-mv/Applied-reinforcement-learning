import torch
from torch import nn


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.ELU()
        )

    def forward(self, x):
        return self.model(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.ELU()
        )

    def forward(self, x):
        return self.model(x)


class CnnEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, downsamplings):
        super().__init__()
        modules = [nn.Conv2d(in_channels, hidden_channels, 5, 1, 2), nn.ELU()]
        for i in range(downsamplings):
            modules.append(DownsamplingBlock(hidden_channels * 2**i, hidden_channels * 2**(i+1)))
        modules.append(nn.Conv2d(hidden_channels * 2**downsamplings, out_channels, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MeanReduce(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(-1, -2))


class MaxReduce(nn.Module):
    def forward(self, x):
        return x.view(*x.shape[:-2], -1).max(-1)[0]


class MeanMaxReduce(nn.Module):
    def forward(self, x):
        left = torch.mean(x, dim=(-1, -2))
        right = x.view(*x.shape[:-2], -1).max(-1)[0]
        return torch.cat([left, right], dim=-1)


class DenseDecoder(nn.Module):
    def __init__(self, input_w, input_h, input_c, reduce_type, hidden_size, output_size):
        """
        :param input_w: Ширина входного тензора. Используется только если `reduce_type == 'flatten'`
        :param input_h: Высота входного тензора. Используется только если `reduce_type == 'flatten'`
        :param input_c: Количество каналов во входном тензоре
        :param reduce_type: Способ превращения многомерного тензора в одномерный
        :param hidden_size: Размер скрытого слоя
        :param output_size: Размер выходного слоя
        """
        super().__init__()
        if reduce_type == "flatten":
            start_features = input_w * input_h * input_c
            self.reduce = nn.Flatten()
        elif reduce_type == "mean":
            start_features = input_c
            self.reduce = MeanReduce()
        elif reduce_type == "max":
            start_features = input_c
            self.reduce = MaxReduce()
        elif reduce_type == "mean_max":
            start_features = input_c * 2
            self.reduce = MeanMaxReduce()
        self.model = nn.Sequential(
            nn.Linear(start_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, img_obs):
        return self.model(self.reduce(img_obs))