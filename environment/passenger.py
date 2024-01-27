class Passenger():
    def __init__(self, start_floor, end_floor, arrival_time, passenger_id, direction):
        self.wait_time = 0.0
        self.floor = start_floor
        self.end_floor = end_floor
        self.arrival_time = arrival_time
        self.passenger_id = passenger_id
        self.direction = direction

    def __eq__(self, other):
        if isinstance(other, Passenger):
            return other.passenger_id == self.passenger_id
        return False

    def __hash__(self):
        return hash(self.passenger_id)

    def __str__(self):
        return f"Passenger id: {self.passenger_id}"

    def __repr__(self):
        return repr(f"Passenger id: {self.passenger_id}")

    def update_floor(self, floor):
        self.floor = floor
