import collections


class Aircraft(collections.namedtuple('Aircraft', ['id', 'name', 'cruise_speed_kts'])):
    KTS_TO_M_PER_S = 0.51444

    def get_max_distance_m(self, episode_time_s: float) -> float:
        """ Estimates the maximum distance this aircraft can travel in an episode """
        margin = 0.2
        return self.cruise_speed_kts * self.KTS_TO_M_PER_S * episode_time_s * (1 + margin)


cessna172P = Aircraft('c172p', 'Cessna172P', 120)
f15 = Aircraft('f15', 'F15', 780)   # cruise speed at low altitude
a320 = Aircraft('A320', 'A320', 490)
