import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.to(self.device)

    def act(self, state):
        state = torch.tensor(state, device=self.device)
        
        return self.model(state).argmax(-1).cpu().numpy()
