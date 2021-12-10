import torch
import random
import numpy as np
import abc
from torch import nn
import copy


class Policy(nn.Module):
    @abc.abstractmethod
    def act(self, img_observation):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class Agent:
    def __init__(self, policy: Policy, device: str):
        self.policy = copy.deepcopy(policy)
        self.policy.cpu()
        self.device = device

    def act(self, observation):

        with torch.no_grad():
            img_observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy.act(img_observation)
        return action.cpu().numpy()[0]

    def update_policy_device(self):
        self.policy.to(self.device)

    def set_policy(self, policy):
        self.policy = policy
        self.policy.to(self.device)


class ExplorationAgent:
    def __init__(self, policy: Policy, device: str, exploration_config: dict):
        self.policy = copy.deepcopy(policy)
        self.policy.cpu()
        self.device = device
        self.max_random_action_proba = exploration_config.get("max_random_action_probability", None)
        self.min_random_action_proba = exploration_config.get("min_random_action_probability", None)
        self.decay_steps = exploration_config.get("decay_steps", None)
        self.action_size = exploration_config["action_size"]
        self.step = 0

    def act(self, observation):
        self.step += 1
        step_coef = self.step / self.decay_steps if self.decay_steps is not None else 0.
        if self.max_random_action_proba is not None and self.min_random_action_proba is not None:
            random_action_prob = self.max_random_action_proba + (self.min_random_action_proba - self.max_random_action_proba) * step_coef
        else:
            random_action_prob = 0.

        if random_action_prob > random.random():
            action = random.randint(0, self.action_size-1)
        else:
            with torch.no_grad():
                img_observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.policy.act(img_observation)
                action = action.cpu().numpy()[0]
        return action

    def update_policy_device(self):
        self.policy.to(self.device)

    def set_policy(self, policy):
        self.policy = policy
        self.policy.to(self.device)
