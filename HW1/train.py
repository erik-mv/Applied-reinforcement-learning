from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 1000000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
HIDDEN_SIZE = 128


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nn.Sequential(
            nn.Linear(state_dim, int(HIDDEN_SIZE / 2)),
            nn.ReLU(),
            nn.Linear(int(HIDDEN_SIZE / 2), HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE / 2)),
            nn.ReLU(),
            nn.Linear(int(HIDDEN_SIZE / 2), action_dim)
        )
        self.target_model = copy.deepcopy(self.model)
        
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.replay_buffer = deque(maxlen=INITIAL_STEPS)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        state, action, next_state, reward, done = transition

        action_ = [0 for i in range(8)]
        action_[0] = action

        reward_ = [0 for i in range(8)]
        reward_[0] = reward

        done_ = [0 for i in range(8)]
        if done:
            done_[0] = 1

        self.replay_buffer.append([state, action_, next_state, reward_, done_])

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        batch = np.array(batch, dtype=np.float32).reshape(BATCH_SIZE, 5, -1)
        batch = torch.from_numpy(batch).cuda()
        return batch
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        state = batch[:, 0]
        action = batch[:, 1, 0].type(torch.LongTensor).cuda()
        next_state = batch[:, 2]
        reward = batch[:, 3, 0]
        done = batch[:, 4, 0].type(torch.LongTensor).cuda()

        q_target = torch.zeros(reward.size()[0]).float()

        with torch.no_grad():
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target[done == 1] = 0

        q_target = (reward + q_target * GAMMA).unsqueeze(1)

        q_model = self.model(state).gather(1, action.unsqueeze(1))

        loss = F.mse_loss(q_model, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.tensor(state).cuda()
        state = state.float().unsqueeze(0)
        action = self.model(state)[0].max(0)[1].view(1, 1).item()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self, mean_rewards, std_rewards):
        torch.save(
            self.model, 
            "agent_mean_" + str(int(mean_rewards)) + "_std_" + str(int(std_rewards)) + ".pkl"
            )


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    mean_rewards = 0
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS// 200) == 0:
            print(f"Step: {i+1}")

        if (i + 1) % 1000 == 0:
            rewards = evaluate_policy(dqn, 5)
            new_mean_rewards = np.mean(rewards)
            if mean_rewards < new_mean_rewards:
                std_rewards = np.std(rewards)
                if std_rewards < 20:
                    mean_rewards = new_mean_rewards
                    print(f"Step: {i+1}, Reward mean: {mean_rewards}, Reward std: {std_rewards}")
                    dqn.save(mean_rewards, std_rewards)
