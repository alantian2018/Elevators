import random
import simpy
import numpy as np
class Elevator():
    def __init__(self, env, id, nfloors,  floor_time = 1.45, stop_time = 3.6, turn_time = 1, capacity = 20, initial_floor = 1):
        self.building = env
        self.id = id
        self.floor_time = floor_time
        self.stop_time = stop_time
        self.in_between_time = stop_time * 1.6
        self.turn_time = turn_time
        self.max_capacity = capacity
        self.floor = initial_floor
        self.passengers = set()
        self.dest_floors = set()
        self.current_capacity = 0
        self.nfloors = nfloors
        self.last_decision_epoch = self.building.now()
        self.output = [i for i in range (10)]
        self.actions = {
            0 : self._move_move,
            1 : self._move_idle,

            2 : self._idle_up_move,
            3 : self._idle_up_step,
            4 : self._idle_down_move,
            5 : self._idle_down_step,
            6 : self._idle_idle,

            7 : self._idle_intent_up,
            8 : self._idle_intent_down,
            9 : self._idle_intent_idle

        }

      #  self.action_space = len(self.get_states(True))
        self.building.simenv.process(self.action_event(6))

        """
        Elevator theory crafting...
        
        3 possible states = up, down, idle
        
        3 possible intentions = up, down, idle
        
        Note that intentions are limited by state. 
            Moving up = cannot have down intent, must first idle -> intent to go down
            
        Thus, we have the following
            
            State     Intent
                 ---- up (keep moving)
                /
             up  
                \
                 ---- idle (stop)
                 
                 
                  ---- down (keep moving)
                 /
             down  
                 \
                  ---- idle (stop)
             
             
                  ---- up (move up)
                 /---- up1 (move up 1)
             idle ---- idle (dont move)
                 \---- down1 (move down1)
                  ---- down (move down) 
             
        Why do we need intentions?
        Idea is that they convey information info. 
        ie, it takes time for instructions to be executed,
        and thus we need to store intent in simulation.
        
        See this great repo for elevator implementation:
        https://github.com/sy2737/Elevator-MARL/blob/master/environment/elevator.py
        """

        self.MOVE_UP = 1
        self.MOVE_DOWN = -1
        self.IDLE = 0

        self.INTENT_NOT_SET = -2
        self.INTENT_UP = 1
        self.INTENT_DOWN = -1
        self.INTENT_IDLE = 0

        self.state = self.IDLE
        self.intent = self.INTENT_NOT_SET
    def __str__(self):
        return (f"Elevator {self.id} at floor {self.floor} with {self.current_capacity} people")

    def interrupt_elevator(self):
        assert self.intent == self.INTENT_NOT_SET and self.state==self.IDLE
        self.idling_event.interrupt()
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def action_event(self, action):
        #print(f"Elevator currently at floor {self.floor}")
        if action == 6:
            self.idling_event = self.building.simenv.process(self.actions[action]())
            try:
                yield self.idling_event
            except simpy.Interrupt:
                self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

        else:
          #  print(f"My action {action}")
            yield self.building.simenv.process(self.actions[action]())
    def _update_floor(self):
        self.floor += self.state
        for i in self.passengers:
            i.update_floor(self.floor)

    def _move_move(self):
        # moving -> moving

        self.intent = self.INTENT_NOT_SET
        yield self.building.simenv.timeout(self.floor_time)
        self._update_floor()
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _move_idle(self):
        # moving -> idle
        self.intent = self.INTENT_NOT_SET
        yield self.building.simenv.timeout(self.stop_time)
        self._update_floor()
        self.state = self.IDLE
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_up_move(self):
        # idle -> up move
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVE_UP

        yield self.building.simenv.timeout(self.stop_time)
        self._update_floor()
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_up_step(self):
        # move up by 1
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVE_UP
        self._update_floor()
        yield self.building.simenv.timeout(self.in_between_time)
        self.state = self.IDLE
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_down_move(self):
        # moving -> down move
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVE_DOWN

        yield self.building.simenv.timeout(self.stop_time)
        self._update_floor()
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_down_step(self):
        # move down by 1
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVE_DOWN
        self._update_floor()
        yield self.building.simenv.timeout(self.in_between_time)
        self.state = self.IDLE
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_idle(self):
        self.intent = self.INTENT_NOT_SET
        assert self.state == self.IDLE
        # Stay idle for at most sometime, and then decide if it wants to stay idle again
        yield self.building.simenv.timeout(10)
        self.building.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_intent_up(self):
        self.intent = self.INTENT_UP
        yield self.building.loading_event(self)
        self.building.trigger_epoch_event("LoadingFinished_{}".format(self.id))

    def _idle_intent_down(self):
        self.intent = self.INTENT_DOWN
        yield self.building.loading_event(self)
        self.building.trigger_epoch_event("LoadingFinished_{}".format(self.id))

    def _idle_intent_idle(self):
        self.intent = self.INTENT_IDLE
        yield self.building.loading_event(self)
        self.building.trigger_epoch_event("LoadingFinished_{}".format(self.id))
    def legal_actions(self):
        legal = set()
        if self.state == self.IDLE:
            if self.intent == self.INTENT_NOT_SET:

                legal.update([7, 8, 9])
                if self.floor == 1:
                    legal.remove(8)
                if self.floor == self.nfloors:
                    legal.remove(7)
            else:
                # intent was declared, so you can move!
                if self.intent == self.INTENT_UP:
                    # note that for _move_move, you must move up at least 2 floors before stopping
                    legal.update([2, 3])
                    # If almost at the top, you have to stop at the next floor up
                    if self.floor == self.building.floors - 1:
                        legal.remove(2)
                elif self.intent == self.INTENT_DOWN:
                    legal.update([4, 5])
                    # If almost at the bottom, you have to stop at the next floor below
                    if self.floor == 2:
                        legal.remove(4)
                else:
                    legal.update([6])
        else:
            legal.update([0,1])
            if self.floor == self.building.floors-1 and self.state==self.MOVE_UP:
                legal.remove(0)
            if self.floor == 2 and self.state==self.MOVE_DOWN:
                legal.remove(0)
      #  if (9 in legal and 0!=len(self.building.active_passengers)):
       #     legal.remove(9)
        return legal

    def unload_passengers(self, floor, building):
        #print('Huh, it actually works')
        people = [p for p in self.passengers]
        time_waited = 0.0

        if floor in self.dest_floors:
            self.dest_floors.remove(floor)

        for p in people:
            if floor == p.end_floor:
                self.passengers.remove(p)
                building.active_passengers.remove(p)
                building.number_successful_passengers += 1
                self.current_capacity -= 1
                time_waited += self.building.now() - p.arrival_time
        return time_waited


    def load_person(self, person):
        if self.current_capacity != self.max_capacity:
            self.passengers.add(person)
            self.current_capacity += 1
            self.dest_floors.add(person.end_floor)
            return True
        return False

    def _one_hot_encode(self, value, dimension):
       # print(value,dimension)
        output = np.zeros(dimension)
        output[value] = 1

        return output

    def get_states(self, decision_epoch):

        elevator_positions = [self.floor] + [e.floor for e in self.building.elevators if e is not self]

        onehot_elevator_positions = np.concatenate([
            self._one_hot_encode(fl - 1, self.nfloors)  for fl in elevator_positions
        ])

        elevator_states = [self.state] + [e.state for e in self.building.elevators if e is not self]

        onehot_elevator_states = np.concatenate([
            self._one_hot_encode(
                state+1, 3
            ) for state in elevator_states
        ])

        requested_calls = [False] * (self.building.floors+1)
        for fl in self.dest_floors:
            #print(fl)
            requested_calls[fl] = True
        time_elapsed = [self.building.now()-self.last_decision_epoch]
        state_representation = np.concatenate([

            self.building.hall_call_up,
            self.building.hall_call_down,
            onehot_elevator_positions,
            onehot_elevator_states,
            requested_calls,
            [self.current_capacity],
            time_elapsed
        ])
        #assert len(state_representation)==self.env.observation_space_size, "should probably modify the obs_space in env.py to match the state output of Elevator.get_states()"
        if decision_epoch:
            self.last_decision_epoch = self.building.simenv.now
   #     print(state_representation)
        return state_representation