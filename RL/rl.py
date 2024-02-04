import sys
import torch
import numpy as np
from nnet import *
import matplotlib.pyplot as plt
sys.path.insert(1, "C:\\Users\\alant\\Python_Projects\\pythonProject\\Elevators\\environment")
import matplotlib.pyplot as plt
from environment.building import start
floors = 10
elevators = 1
building = start(floors,elevators)
observation_space_size = len(building.elevators[0].get_states(True))
action_space_size = 10
print(f"Observation space size: {observation_space_size}")
sim_hours = 100
time_graph = []
Agents = []
optimizers = []
verbose =True
last_checkpoint_time = 0.0
for i in range (elevators):
    Agents.append(
        DQN(observation_space_size, action_space_size)
    )
    optimizers.append(torch.optim.Adam(Agents[i].parameters(), lr=0.001))

wait_time_list = []
env_vars = building.reset() # get states

def select_argmax(tensor):
    m=-float('inf')
    ind = 0
    for c,i in enumerate(tensor):
        if i>m and i!=0.0:
           ind = c
           m=i
    return ind


while sim_hours * 3600 >= building.now():


    states = env_vars[0]
    #print("What a fucking miracle")
    decision_agents = env_vars[-1]
    R = env_vars[1]

    actions = []

    # each decision agent updates NNet and selects action
    for i in range(len(decision_agents)):
        with torch.enable_grad():
            agent = decision_agents[i]

          #  print(agent)
          #  print(f'R {R}')

            # construct legal_actions_list, in ascending order of action index
            legal_actions_list = list(building.elevators[agent].legal_actions())
            legal_actions_list.sort()
            # construct a boolean representation of legal_actions
            legal_actions_bool = np.full(action_space_size, False)
            for action in building.elevators[agent].legal_actions():
                legal_actions_bool[action] = True
            actions = []
            #print(f'Legal Actions {legal_actions_list}')
            current_weights = select_action(Agents[agent], states[agent], legal_actions_bool)

            #print(f'Probability actions: {current_weights}')
            #print(agent)
            #print(current_weights)
            actions.append(select_argmax(current_weights))
            Q_value  = current_weights + R[agent]
            #print(Q_value)
            #Q_value = torch.tensor(Q_value)
            Q_value = torch.tensor(Q_value)

            criterion = nn.SmoothL1Loss()
            loss = criterion(current_weights, Q_value)
            loss.requires_grad = True
            #print(f'Loss {loss}')

            # Optimize the model
            loss.backward()

            optimizers[agent].step()
            optimizers[agent].zero_grad()
    if building.now() - last_checkpoint_time > 5 * 3600:

        print(f'Hour: {round(building.now()//3600)}')
        print(f'Loss: {loss}')
        print(f"Number of people processed: {building.number_successful_passengers}")
        if building.number_successful_passengers!=0:
            print(f'Average waited time: {building.total_time / building.number_successful_passengers}')
            time_graph.append(building.total_time / building.number_successful_passengers)
        print(f"Total ppl waiting: {len(building.active_passengers)}")
        print(f"Legal actions: {legal_actions_list}. Current floor: {building.elevators[agent].floor}")
        print('\n\n')
        last_checkpoint_time = building.now()



    env_vars = building.step(actions)
    #print (building.number_successful_passengers)
plt.plot(time_graph)
plt.show()