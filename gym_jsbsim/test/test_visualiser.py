import time
import unittest
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.visualiser import FlightGearVisualiser


class TestFlightGearVisualiser(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.flightgear = None
        # individual test methods should init as needed:
        # self.flightgear = FlightGearVisualiser(self.sim)

    def tearDown(self):
        self.sim.close()
        if self.flightgear:
            self.flightgear.close()

    def test_launch_flightgear(self):
        self.flightgear = FlightGearVisualiser(self.sim,
                                               block_until_loaded=False)
        time.sleep(0.5)

        # check FlightGear has launched by looking at stdout
        self.assertIn('FlightGear', self.flightgear.flightgear_process.stdout.readline().decode())
        self.flightgear.close()

    def test_close_closes_flightgear(self):
        self.flightgear = FlightGearVisualiser(self.sim,
                                               block_until_loaded=False)
        self.flightgear.close()
        return_code = self.flightgear.flightgear_process.wait()
        # a non-None return code indicates termination
        self.assertIsNotNone(return_code)
