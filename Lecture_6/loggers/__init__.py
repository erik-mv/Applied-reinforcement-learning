import abc


class Logger:
    @abc.abstractmethod
    def add_value(self, key, value):
        pass

    @abc.abstractmethod
    def set_step(self, step):
        pass

    @abc.abstractmethod
    def next_step(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass