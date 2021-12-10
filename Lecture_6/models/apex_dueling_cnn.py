import torch
from torch import nn
from torch.nn import functional
from .blocks import CnnEncoder, DenseDecoder


class ApexDuelingArchitecture(nn.Module):
    def __init__(self, in_channels, num_actions, num_downsamplings=4, hidden_channels=8, hidden_features=256):
        """
        :param in_channels: Количество каналов в части наблюдения, соответствующей двумерным картам признаков
        :param num_actions: Количество действий
        :param num_downsamplings: Количество downsampling block'ов, которые будут применены к состоянию
        :param hidden_channels: Количество скрытых каналов в начале свертки
        :param hidden_features: Количество скрытых нейронов в полносвязной части сети
        """
        super().__init__()
        bottleneck_channels = hidden_channels * 2**num_downsamplings  # Количество каналов в выходе Encoder'а
        self.encoder = CnnEncoder(in_channels, hidden_channels, bottleneck_channels, num_downsamplings)
        self.value = DenseDecoder(-1, -1, bottleneck_channels, "mean_max", hidden_features, 1)
        self.advantage = DenseDecoder(-1, -1, bottleneck_channels, "mean_max", hidden_features, num_actions)

    def get_q(self, img_observation):
        x = self.encoder(img_observation)
        adv = self.advantage(x)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        value = self.value(x).expand_as(adv)
        return adv + value

    def get_adv(self, img_observation):
        x = self.encoder(img_observation)
        adv = self.advantage(x)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        return adv