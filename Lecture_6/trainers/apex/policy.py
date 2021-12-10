import torch
from agent.agent import Policy
from torch import nn
from models.apex_dueling_cnn import ApexDuelingArchitecture


class ApexPolicy(Policy):
    def __init__(self, model_config):
        """
        :param model_config: Конфиг для модели DQN
        """
        super().__init__()
        self.model = ApexDuelingArchitecture(**model_config)

    def act(self, img_observation):
        """
        :param img_observation: Часть наблюдения, соответствующая двумерным картам признаков
        :return: Действие, являющееся целым числом
        """
        # Для того, чтобы вырать лучшее действие, можно использовать только Advantage
        adv = self.model.get_adv(img_observation)
        return adv.argmax(-1)

    def get_q(self, img_observation):
        """
        :param img_observation:  Часть наблюдения, соответствующая двумерным картам признаков
        :return: Действие, являющееся целым числом
        """
        return self.model.get_q(img_observation)

    def save(self, path):
        """
        :param path: Путь, по которому будет сохранен чекпоинт
        """
        torch.save(self.model.state_dict(), f"{path}/dqn.pkl")

    def load(self, path):
        """
        :param path: Путь, по которому расположен необходимый чекпоинт
        """
        self.model.load_state_dict(torch.load(f"{path}/dqn.pkl"))
