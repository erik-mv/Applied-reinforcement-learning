from . import Logger
from .json_logger import JsonLogger, JsonStateLogger
from .tensorboard_logger import TensorboardLogger
import json
import os
import numpy as np


class JointLogger(Logger):
    def __init__(self, path, config):
        """
        Логгер, который предоставляет универсальный интерфейс для сохранения данных в TensorBoard, JSON,
        а также соранения чекпоинтов агента
        :param path: Путь, по которому будут сохранены логи
        :param config: Конфиг модели, который будет сохранен вместе с логами
        """
        os.makedirs(path, exist_ok=True)
        json.dump(config, open(f"{path}/config.json", "w"))
        self.path = path
        self.json_logger = JsonLogger(f"{path}/log.json")
        self.tb_logger = TensorboardLogger(f"{path}/tb_log")
        self.step = 0

    def add_value(self, key, value):
        self.json_logger.add_value(key, value)
        self.tb_logger.add_value(key, value)

    def set_step(self, step):
        self.json_logger.set_step(step)
        self.tb_logger.set_step(step)
        self.step = step

    def next_step(self):
        self.json_logger.next_step()
        self.tb_logger.next_step()

    def save_agent(self, policy):
        path = f"{self.path}/checkpoints/{self.step}"
        os.makedirs(path, exist_ok=True)
        policy.save(path)

    def close(self):
        self.json_logger.close()
        self.tb_logger.close()