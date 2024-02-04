import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations+10, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(net, state, legal_actions):
    with torch.no_grad():
        state = torch.tensor(state)
        state = state.to(torch.float32)
        legal_actions = torch.tensor(legal_actions)
        legal_actions = legal_actions.to(torch.float32)
        states = torch.cat([state, legal_actions])
        weights = net(states)
        weights *= legal_actions
        return weights