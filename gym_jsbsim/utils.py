import math
from typing import Tuple


class LatLonPosition(object):
    FT_PER_DEG_LAT = 365228
    EARTH_RADIUS_FT = 2.0902e+7

    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = latitude_deg
        self.lon = longitude_deg

    def _calculate_ft_per_deg_lon(self, longitude_deg: float) -> float:
        # ft per deg. longitude is distance at equator * cos(lon)
        # attribution: https://www.colorado.edu/geography/gcraft/warmup/aquifer/html/distance.html
        return self.FT_PER_DEG_LAT * math.cos(math.radians(longitude_deg))

    def heading_to(self, destination: 'LatLonPosition') -> float:
        """ Determines heading in degrees of course between self and destination """
        # formula attribution:
        #   https://www.movable-type.co.uk/scripts/latlong.html
        _, delta_lon = destination - self
        y = math.sin(delta_lon) * math.cos(destination.lat)
        x = (math.cos(self.lat) * math.sin(destination.lat) -
             math.sin(self.lat) * math.cos(destination.lat) * math.cos(delta_lon))

        heading_rad = math.atan2(y, x)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    def distance_to(self, destination: 'LatLonPosition') -> float:
        """ Determines distance in feet to destination

        Appropriate only for small differences in lat/long because it uses the
        equirectangular projection (doesn't account for Earth being sphere).
        """
        # formula based on equirectangular distance formula given by:
        #   https://www.movable-type.co.uk/scripts/latlong.html
        delta_lat, delta_lon = destination - self

        mean_lat = (self.lat + destination.lat) / 2
        delta_lon_corrected = delta_lon * math.cos(mean_lat)

        return self.EARTH_RADIUS_FT * ((delta_lat ** 2 + delta_lon_corrected ** 2) ** 0.5)

    def distance_parallel_to_heading_ft(self, finish: 'LatLonPosition', desired_heading_deg: float) -> float:
        """ Determines distance in feet travelled parallel to the desired heading """
        pass

    def __sub__(self, other) -> Tuple[float, float]:
        """ Returns difference between two coords as (delta_lat, delta_lon) """
        return self.lat - other.lat, self.lon - other.lon
