from . import Logger
import json
import os


class JsonLogger(Logger):
    def add_value(self, key, value):
        if self.step not in self.logs:
            self.logs[self.step] = {}
        logs = self.logs[self.step]
        if key not in logs:
            logs[key] = []
        logs[key].append(value)

    def set_step(self, step):
        self.step = step

    def next_step(self):
        self.step += 1

    def close(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.logs, open(self.path, "w"))

    def __init__(self, path):
        self.path = path
        self.step = 0
        self.logs = {}


class JsonStateLogger:
    def __init__(self, path):
        self.path = path
        self.step = 0
        self.logs = {}

    def add_states(self, states):
        if self.step not in self.logs:
            self.logs[self.step] = []
        self.logs[self.step].extend(states)

    def close(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.logs, open(self.path, "w"))

    def set_step(self, step):
        self.step = step