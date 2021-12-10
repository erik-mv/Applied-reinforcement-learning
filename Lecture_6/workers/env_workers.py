import gym
from torch import multiprocessing as mp
import numpy as np
import copy
import cv2
from collections import deque


class EnvWrapper:
    def __init__(self, env, config):
        """
        :param env: Окружение
        :param config: Конфиг, который описывает способ преобразования наблюдения
        """
        self.env = env
        self.stack_frames = config["stack_frames"]
        self.last_frames = deque(maxlen=self.stack_frames)

    def _transform_obseration(self, observation):
        observation = cv2.resize(observation, (64, 64))
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        while len(self.last_frames) < self.stack_frames:
            self.last_frames.append(observation)
        self.last_frames.append(observation)
        return np.stack(self.last_frames, axis=0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self._transform_obseration(state)
        return state, reward, done, info

    def reset(self):
        self.last_frames.clear()
        return self._transform_obseration(self.env.reset())


def env_worker(agent, env_name, wrapper_config, transitions_queue: mp.Queue, policy_queue: mp.Queue):
    env = EnvWrapper(gym.make(env_name), wrapper_config)
    agent.update_policy_device()
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        transitions_queue.put((state, action, next_state, reward, done))
        if done:
            state = env.reset()
        else:
            state = next_state

        new_agent_policy = None
        try:
            while True:
                new_agent_policy = policy_queue.get(block=False)
        except:
            pass
        if new_agent_policy is not None:
            agent.set_policy(new_agent_policy)


class EnvPool:
    def __init__(self, agent, env_name, wrapper_config, n_workers):
        self.transition_queue = mp.Queue(maxsize=3*n_workers)
        self.agent_policy_queues = [mp.Queue(maxsize=2) for _ in range(n_workers)]
        self.workers = [mp.Process(target=env_worker, args=(agent, env_name, wrapper_config, self.transition_queue, pq))
                        for pq in self.agent_policy_queues]
        for w in self.workers:
            w.start()

    def collect_experience(self, transitions_number):
        result = [self.transition_queue.get() for _ in range(transitions_number)]
        return result

    def update_agent(self, policy):
        policy = copy.deepcopy(policy)
        policy.cpu()
        for q in self.agent_policy_queues:
            q.put(policy)

    def close(self):
        for w in self.workers:
            w.terminate()
        for w in self.workers:
            w.join()
