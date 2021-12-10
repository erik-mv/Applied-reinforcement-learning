import numpy as np
import torch
from collections import deque
import time
import random
from utils.sum_tree import SumTree


def build_buffer(buffer_config):
    """
    Метод создает один из вариантов буфера опыта, используя конфиг. Все имплементации буфера имеют одинаковый интерфейс
    :param buffer_config: Конфиг буфера
    :return: Буфер опыта, созданный в соответствии с конфигом
    """
    if buffer_config["name"] == "uniform_replay":
        return ExperienceReplay(buffer_config)
    elif buffer_config["name"] == "prioritized_replay":
        return PrioritizedExperienceReplay(buffer_config)

class ExperienceReplay:
    def __init__(self, config):
        self.size = config["size"]
        self.memory = deque(maxlen=self.size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size, progress):
        batch = [self.memory[random.randint(0, len(self.memory)-1)] for _ in range(batch_size)]
        weights = np.ones(batch_size)
        info = None
        batch = list(zip(*batch))
        return batch, weights, info

    def update_priority(self, info, error):
        pass


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, config):
        super().__init__(config)
        self.sum_tree = SumTree(self.size)
        self.max_priority = 1.
        self.beta_start = config["beta_start"]
        self.beta_end = config["beta_end"]
        self.alpha = config["alpha"]

    def append(self, transition):
        self.sum_tree.add(self.max_priority, transition)

    def sample(self, batch_size, progress):
        beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        batch = []
        weights = []
        info = []
        for _ in range(batch_size):
            p = random.random()
            idx, transition, weight = self.sum_tree.get(p)
            info.append(idx)
            batch.append(transition)
            weights.append((self.sum_tree.n_entries * weight)**-beta)  # Importance Sampling
        batch = list(zip(*batch))
        weights = np.array(weights) / np.sum(weights)  # Теперь сумма весов равна 1
        return batch, weights, info

    def update_priority(self, info, error):
        priority = (np.array(error)**self.alpha + 1e-2)
        self.max_priority = max(self.max_priority*0.99, np.max(priority), 1.)
        for idx, p in zip(info, priority):
            self.sum_tree.update(idx, p)