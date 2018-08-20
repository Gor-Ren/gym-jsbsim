import math
import gym_jsbsim.properties as prp
from typing import Tuple
from gym_jsbsim.simulation import Simulation


class GeodeticPosition(object):
    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = latitude_deg
        self.lon = longitude_deg

    def heading_deg_to(self, destination: 'GeodeticPosition') -> float:
        """ Determines heading in degrees of course between self and destination """
        delta_lat, delta_lon = destination - self
        heading_rad = math.atan2(delta_lon, delta_lat)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    @staticmethod
    def from_sim(sim: Simulation) -> 'GeodeticPosition':
        """ Return a GeodeticPosition object with lat and lon from simulation """
        lat_deg = sim[prp.lat_geod_deg]
        lon_deg = sim[prp.lng_geoc_deg]
        return GeodeticPosition(lat_deg, lon_deg)

    def __sub__(self, other) -> Tuple[float, float]:
        """ Returns difference between two Cartesian coords as (delta_lat, delta_long) """
        return self.lat - other.lat, self.lon - other.lon


