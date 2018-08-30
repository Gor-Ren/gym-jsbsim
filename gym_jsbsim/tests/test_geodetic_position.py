from gym_jsbsim.properties import GeodeticPosition
import unittest


class TestGeodeticPosition(unittest.TestCase):
    BATH_LAT = 51.3751
    BATH_LNG = -2.36172
    # we have to accept small loss of accuracy (max +-1 degree) due to using Cartesian projection
    heading_accuracy_places = 0

    def setUp(self, lat=BATH_LAT, lng=BATH_LNG):
        self.position = GeodeticPosition(lat, lng)

    def test_heading_deg_to_north_and_south(self):
        lat, lng = self.BATH_LAT, self.BATH_LNG
        north_lat = lat + 1
        south_position = GeodeticPosition(lat, lng)
        north_position = GeodeticPosition(north_lat, lng)

        north_heading_deg = south_position.heading_deg_to(north_position)
        south_heading_deg = north_position.heading_deg_to(south_position)

        self.assertAlmostEqual(0, north_heading_deg, places=self.heading_accuracy_places)
        self.assertAlmostEqual(180, south_heading_deg, places=self.heading_accuracy_places)

    def test_heading_deg_to_east_and_west(self):
        lat, lng = self.BATH_LAT, self.BATH_LNG
        east_lng = lng + 1
        west_position = GeodeticPosition(lat, lng)
        east_position = GeodeticPosition(lat, east_lng)

        east_heading_deg = west_position.heading_deg_to(east_position)
        west_heading_deg = east_position.heading_deg_to(west_position)

        self.assertAlmostEqual(90, east_heading_deg, places=self.heading_accuracy_places)
        self.assertAlmostEqual(270, west_heading_deg, places=self.heading_accuracy_places)

    def test_heading_deg_ne_and_sw(self):
        lat, lng = self.BATH_LAT, self.BATH_LNG
        north_lat, east_lng = lat + 1, lng + 1
        north_east_position = GeodeticPosition(north_lat, east_lng)
        south_west_position = GeodeticPosition(lat, lng)

        north_east_heading_deg = south_west_position.heading_deg_to(north_east_position)
        south_west_heading_deg = north_east_position.heading_deg_to(south_west_position)

        self.assertAlmostEqual(45, north_east_heading_deg, places=self.heading_accuracy_places)
        self.assertAlmostEqual(225, south_west_heading_deg, places=self.heading_accuracy_places)
