import math
from typing import Tuple
import pyproj


class Position(object):
    CARTESIAN_COORD_ID = 'epsg:27700'  # British National Grid cartesian coord system
    cartesian_projection = pyproj.Proj(init=CARTESIAN_COORD_ID)

    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = latitude_deg
        self.lon = longitude_deg
        # calculate Cartesian x,y coord
        self.x_m, self.y_m = self.cartesian_projection(self.lon, self.lat)

    def heading_deg_to(self, destination: 'Position') -> float:
        """ Determines heading in degrees of course between self and destination """
        delta_x, delta_y = destination - self
        heading_rad = math.atan2(delta_x, delta_y)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    def distance_m_to(self, destination: 'Position') -> float:
        """ Determines distance in metres to destination """
        delta_x, delta_y = destination - self
        return math.hypot(delta_x, delta_y)

    def distance_parallel_to_heading_ft(self, finish: 'Position', desired_heading_deg: float) -> float:
        """ Determines distance in feet travelled parallel to the desired heading """
        total_distance_m = self.distance_m_to(finish)
        actual_heading_deg = self.heading_deg_to(finish)
        heading_error_rad = math.radians(actual_heading_deg - desired_heading_deg)

        parallel_distance_m = total_distance_m * math.cos(heading_error_rad)
        return parallel_distance_m

    def __sub__(self, other) -> Tuple[float, float]:
        """ Returns difference between two Cartesian coords as (delta_x, delta_y) """
        return self.x_m - other.x_m, self.y_m - other.y_m
