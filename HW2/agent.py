import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        #self.model.to(self.device)
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).unsqueeze(0).float()
            return self.model.act(state)[0].flatten().numpy()


    def reset(self):
        pass

