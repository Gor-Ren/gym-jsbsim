import time
import unittest
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.visualiser import FigureVisualiser, FlightGearVisualiser
from gym_jsbsim.test.stubs import TaskStub

class TestFlightGearVisualiser(unittest.TestCase):
    env = None
    sim = None
    flightgear = None

    def setUp(self):
        if self.env:
            self.env.close()
        if self.sim:
            self.sim.close()
        self.env = JsbSimEnv(task_type=TaskStub)
        self.env.reset()
        self.sim = self.env.sim
        self.flightgear = None
        # individual test methods should init as needed:
        # self.flightgear = FlightGearVisualiser(self.sim)

    def tearDown(self):
        if self.env:
            self.env.close()
        if self.sim:
            self.sim.close()
        if self.flightgear:
            self.flightgear.close()

    def test_init_creates_figure(self):
        self.flightgear = FlightGearVisualiser(self.sim, block_until_loaded=False)
        self.assertIsInstance(self.flightgear.figure, FigureVisualiser)

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
        timeout_seconds = 2.0
        return_code = self.flightgear.flightgear_process.wait(timeout=timeout_seconds)
        # a non-None return code indicates termination
        self.assertIsNotNone(return_code)

    def test_plot_displays_actions(self):
        self.setUp()
        self.flightgear = FlightGearVisualiser(self.sim,
                                               block_until_loaded=False)
        action_names = self.env.task.action_names
        action = self.env.task.get_action_space().sample()
        self.flightgear.plot(self.sim)

        # the figure should have plotted a Lines object each axis
        for axis in ['axes_stick', 'axes_rudder', 'axes_throttle']:
            axis_data_plots = getattr(self.flightgear.figure.axes, axis)
            is_empty_plot = len(axis_data_plots.axes.lines) == 0
            self.assertFalse(is_empty_plot,
                             msg=f'no data plotted on axis {axis}')