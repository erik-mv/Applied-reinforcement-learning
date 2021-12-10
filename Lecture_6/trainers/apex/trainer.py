from .policy import ApexPolicy
from copy import deepcopy
from utils.common import soft_update
from utils.memory import build_buffer
from torch.optim import Adam
import numpy as np

import torch
from torch.nn import functional as F
import copy
import time


class ApexTrainer:
    def __init__(self, policy: ApexPolicy, buffer_config, discount_factor=0.99, batch_size=64,
                 updates_per_consumption=16, soft_update_tau=0.01, device="cuda"):
        """
        :param policy: Нейронная сеть, соответствующая интерфейсу ApexPolicy
        :param buffer_config: Конфиг для буфера опыта
        :param discount_factor: Фактор дисконтирования награды
        :param batch_size: Размер батча для обучения
        :param updates_per_consumption: Количество обновлений сети при одном вызове метода consume_transitions
        :param soft_update_tau: Значение скорости soft_update'а
        :param device: Устройство, на котором будет происходить обучение
        """
        self.policy = policy
        self.target_policy = copy.deepcopy(self.policy)
        self.replay_buffer = build_buffer(buffer_config)
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.updates_per_consumption = updates_per_consumption
        self.tau = soft_update_tau
        self.device = device
        self.optim = Adam(self.policy.parameters(), 5e-4)

    def consume_transitions(self, transitions, progress):
        """
        :param transitions: Набор транзиций, принимаемых на вход методом
        :param progress: Число от 0 до 1, отражающее количество завершенных итераций обучения
        :return: Словарь метрик, которые должны быть добавлены в лог
        """
        for t in transitions:
            self.replay_buffer.append(t)

        for _ in range(self.updates_per_consumption):
            log_info = {}

            ##### Предобработка батча
            batch, weights, buffer_info = self.replay_buffer.sample(self.batch_size, progress)
            state, action, next_state, reward, done = batch

            img_state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_img_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(-1)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            ##### Обновление DQN
            with torch.no_grad():
                target_q = self.target_policy.get_q(next_img_state)
                target_q = target_q.max(-1)[0]
                target_q = reward + self.gamma * (1 - done) * target_q.view(-1)

            q = self.policy.get_q(img_state)
            q = q.gather(-1, action).view(-1)
            errors = F.smooth_l1_loss(q, target_q, reduction="none")  # Как правило стабильнее чем MSE
            loss = (weights * errors).sum()  # Взвешенное среднее
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            soft_update(self.policy, self.target_policy, self.tau)  # Вместо hard update, на практике ускоряет обучение
            self.replay_buffer.update_priority(buffer_info, errors.detach().cpu().numpy())

            log_info["loss"] = loss.detach().cpu().item()

        return log_info