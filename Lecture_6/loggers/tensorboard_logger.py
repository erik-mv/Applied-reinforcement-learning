from tensorboardX import SummaryWriter
import os
from . import Logger


class TensorboardLogger(Logger):
    def add_value(self, key, value):
        self.writter.add_scalar(key, value, self.step)

    def set_step(self, step):
        self.step = step

    def next_step(self):
        self.step += 1

    def close(self):
        self.writter.flush()
        self.writter.close()

    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.writter = SummaryWriter(path)
        self.step = 0