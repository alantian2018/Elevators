import sys
import torch
import numpy as np
from nnet import *
import matplotlib.pyplot as plt
sys.path.insert(1, "C:\\Users\\alant\\Python_Projects\\pythonProject\\Elevators\\environment")
from environment.building import start
floors = 10
elevators = 1
building = start(floors,elevators)
observation_space_size = len(building.elevators[0].get_states(True))
action_space_size = 10
print(f"Observation space size: {observation_space_size}")
sim_hours = 10
Agents = []

optimizers = []
for i in range (elevators):
    Agents.append(
        DQN(observation_space_size, action_space_size)
    )
    optimizers.append(torch.optim.Adam)

wait_time_list = []
env_vars = building.reset() # get states
states = env_vars[0]

while sim_hours * 3600 >= building.now():
    print("What a fucking miracle")
    decision_agents = env_vars[-1]
    R = env_vars[1]
    actions = []

    # each decision agent updates NNet and selects action
    for i in range(len(decision_agents)):
        agent = decision_agents[i]

        # construct legal_actions_list, in ascending order of action index
        legal_actions_list = list(building.elevators[agent].legal_actions())
        legal_actions_list.sort()
        # construct a boolean representation of legal_actions
        legal_actions_bool = np.full(action_space_size, False)
        for action in building.elevators[agent].legal_actions():
            legal_actions_bool[action] = True
        # compute q-values
        loss = None
        print(legal_actions_bool)
