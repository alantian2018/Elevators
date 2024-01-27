import numpy as np
from pyprobs import Probability as pr
from datetime import datetime, timedelta
import math
import simpy
from simpy import AnyOf
import random
from collections import Counter
import matplotlib.pyplot as plt
from elevator import Elevator
from passenger import Passenger
from collections import deque
def start(floors, elevators):
    simenv = simpy.Environment()
    print("making env...")
    return Building(simenv, floors, elevators)




class Building():
    def __init__(self, simenv, floors, elevators):
        """
        :param floors: number of floor in the building
        :param elevators: number of elevators in the building
        :param time_start: time to start sim
        :param timeframe:  amount of time to run sim for
        """
        self.simenv = simenv
        self.floors = floors
        self.people_at_floor = []
        self.nelevators = elevators
        self.directory = [i for i in range (1, floors + 1)]
        self.traffic = []
        self.active_passengers = set()
        self.number_successful_passengers = 0
        self.cost = 0
        self.elevators = [Elevator(self, id, self.floors) for id in range(self.nelevators)]

        self.reset()


    def step(self, actions):
        for idx, a in enumerate(actions):
            if a == -1:
                continue
            self.simenv.process(self.elevators[self.elevators[idx]].act(a))

        while True:
            self.decision_elevators = []
            finished_events = self.simenv.run(until=AnyOf(self.simenv, self.epoch_events.values())).events
            self._calculate_reward()
            # There can be multiple events finished (when passenger arrives and multiple elevators go into decision mode)

            # Here is where we process the events
            # We calculate total waiting time etc, and assign loading events
            # If the event_type qualifies as a decision epoch then break
            # out of the while loop and return the appropriate state
            decision = True
            for event in finished_events:
                event_type = event.value
             #   print(event_type)
                if "PassengerArrival" in event_type:
                    decision = False

                elif "LoadingFinished" in event_type:
                    self.decision_elevators.append(int(event_type.split('_')[-1]))
                    decision = True
                elif 'ElevatorArrival' in event_type:
                    self.decision_elevators.append(int(event_type.split('_')[-1]))
                    decision = True

            if decision:
              #  print('broken')
                break

        return [self.elevators[idx].get_states(True) for idx in range (self.nelevators)], self.get_cost(),self.decision_elevators

    def now(self):
        return self.simenv.now

    def trigger_epoch_event(self, event_type):
        self.epoch_events[event_type].succeed(event_type)
        self.epoch_events[event_type] = self.simenv.event()
    def reset(self):
        self.people_at_floor = [set() for i in range (self.floors+1)]
        self.traffic = []
        self.active_passengers = set()
        self.number_successful_passengers = 0
        self.simenv = simpy.Environment()
        self.simenv.process(self.generate_passengers())
        self.elevators = [Elevator(self, id, self.floors) for id in range(self.nelevators)]
        self.epoch_events = {
            "PassengerArrival": self.simenv.event(),
        }
        for idx in range(self.nelevators):
            self.epoch_events["ElevatorArrival_{}".format(idx)] = self.simenv.event()
        for idx in range(self.nelevators):
            self.epoch_events["LoadingFinished_{}".format(idx)] = self.simenv.event()

        self.hall_call_down = np.array([False]*(1+self.floors))
        self.hall_call_up = np.array([False] * (1+self.floors))

        return self.step([-1])

    def _reward_function(self, time_taken):
        return time_taken ** 2

    def _calculate_reward(self):
        self.cost = 0.0
        for floor in self.people_at_floor:
            for person in floor:
                self.cost += self._reward_function(self.simenv.now - person.arrival_time)

        for ele in self.elevators:
            for person in ele.passengers:
                self.cost += self._reward_function(self.simenv.now - person.arrival_time)


    def take_elevator(self, floors, lobby_weighting = .85):
        """
        :param floors:
        :return: Probability of someone taking the elevator to each floor.
        """
        # 85% chance any given elevator ride will start or end at the lobby

        probabilities = np.empty(floors)

        total = 0
        for i in range (1, floors):

            prob = 2 / (1 + math.exp(- i)) - 1
            total += prob
            probabilities[i] = prob
        probabilities /= total
        probabilities *= (1 - lobby_weighting)
        probabilities[0] = lobby_weighting

        return probabilities

    def visualize_passenger_distribution(self):
        self.traffic = np.asarray(self.traffic)
        fig, ax = plt.subplots(2)

        begins = Counter(self.traffic[:,0])

        ax[0].bar(self.directory, [begins[i] / len(self.traffic) for i in range(1,self.floors + 1)])
        ax[0].set_title("begin on floor")


        ends = Counter(self.traffic[:,1])
        ax[1].bar(self.directory, [ends[i] / len(self.traffic) for i in range(1, self.floors + 1)])
        ax[1].set_title("end on floor")
        # plt.show()

    def loading_event(self, elevator):
        # print(elevator)
        cur_floor = elevator.floor
        self.cost += elevator.unload_passengers(cur_floor, self)
        loaded = 0
        ppl = [p for p in self.people_at_floor[cur_floor]]
        for person in ppl:
            if person.direction == elevator.intent:
                if elevator.load_person(person):
                    self.people_at_floor[cur_floor].remove(person)
                    loaded += 1
                else:
                    break

        if elevator.intent == elevator.INTENT_UP:
            self.hall_call_up[cur_floor] = False
        if elevator.intent == elevator.INTENT_DOWN:
            self.hall_call_down[cur_floor] = False

        return self.simenv.timeout((2 + loaded) * (loaded > 0))

    def generate_passengers(self):
        floor_prob_distr = self.take_elevator(self.floors)
        passenger_id = 0
        while True:
            time = np.random.exponential(350)
            yield self.simenv.timeout(time)

            direction = random.choice([-1,1])
            pth = np.random.choice(self.directory, size = 2, replace = False, p = floor_prob_distr)
            pth.sort()
            if direction == -1:
                pth = pth[::-1]
            arrival_time = self.simenv.now

            passenger = Passenger(pth[0], pth[1], arrival_time, passenger_id, direction)
            self._handle_passenger(passenger, pth)

            # print(f"Arrived: {passenger.start_floor} Dest: {passenger.end_floor} Time: {str(arrival_time)} Timedelta: {str(timedelta(seconds = time))}")

            passenger_id += 1


    def _handle_passenger(self, passenger, pth):

        self.traffic.append(pth)

        self.active_passengers.add(passenger)
        self.people_at_floor[passenger.floor].add(passenger)

        if passenger.direction == -1:
            assert pth[0] != 1
            self.hall_call_down[pth[0]] = True

        else:
            assert pth[0] != self.floors
            self.hall_call_up[pth[0]] = True

        for elevator in self.elevators:
            if elevator.intent == elevator.INTENT_NOT_SET and elevator.state == elevator.IDLE:
            #    print(f'{elevator} interrupted')
                elevator.interrupt_elevator()

        self.trigger_epoch_event("PassengerArrival")
        return False
    def get_cost(self):
        return self.cost
if __name__=='__main__':
    a=start(10,1)
    print(a.elevators[0].get_states(True).shape)
