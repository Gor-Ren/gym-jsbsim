import collections


class Aircraft(collections.namedtuple('Aircraft', ['id', 'cruise_speed_kts'])):
    KTS_TO_M_PER_S = 0.51444

    def get_max_distance_m(self, episode_time_s: float) -> float:
        margin = 1.2  # add 20 % extra
        return self.cruise_speed_kts * self.KTS_TO_M_PER_S * episode_time_s * margin


Cessna172P = Aircraft('c172p', 120)
F15 = Aircraft('f15', 780)   # cruise speed at low altitude
A320 = Aircraft('A320', 490)
