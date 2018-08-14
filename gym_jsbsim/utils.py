import math
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
        lat_deg = sim['position/lat-geod-deg']
        lon_deg = sim['position/long-gc-deg']
        return GeodeticPosition(lat_deg, lon_deg)

    def __sub__(self, other) -> Tuple[float, float]:
        """ Returns difference between two Cartesian coords as (delta_lat, delta_long) """
        return self.lat - other.lat, self.lon - other.lon


def normalise_unbounded_error(absolute_error, error_scaling):
    """
    Given an error in the interval [0, +inf], returns a normalised error in [0, 1]

    The normalised error asymptotically approaches 1 as absolute_error -> +inf.

    The parameter error_scaling is used to scale for magnitude.
    When absolute_error == error_scaling, the normalised error is equal to 0.75
    """
    if absolute_error < 0:
        raise ValueError(f'Error to be normalised must be non-negative '
                         f'(use abs()): {absolute_error}')
    scaled_error = absolute_error / error_scaling
    return (scaled_error / (scaled_error + 1)) ** 0.5
