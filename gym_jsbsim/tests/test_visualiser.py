import time
import unittest
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.visualiser import FigureVisualiser, FlightGearVisualiser
from gym_jsbsim.tests.stubs import BasicFlightTask, DefaultSimStub
import matplotlib.pyplot as plt
import gym_jsbsim.visualiser


class TestFigureVisualiser(unittest.TestCase):
    sim = None
    visualiser = None

    def setUp(self, plot_position=True):
        self.sim = DefaultSimStub()
        task = BasicFlightTask()
        self.visualiser = FigureVisualiser(DefaultSimStub(), task.get_props_to_output())

    def tearDown(self):
        self.visualiser.close()

    def test_plot_creates_figure_and_axes(self):
        self.setUp()

        self.visualiser.plot(self.sim)

        self.assertIsInstance(self.visualiser.figure, plt.Figure)
        self.assertIsInstance(self.visualiser.axes, gym_jsbsim.visualiser.AxesTuple)

    def test_plot_doesnt_plot_position_when_set_by_init(self):
        self.setUp(plot_position=False)

        self.visualiser.plot(self.sim)

        position_axis = self.visualiser.axes.axes_state
        is_empty_plot = position_axis is None or len(position_axis.axes.lines) == 0
        self.assertTrue(is_empty_plot)

    def test_plot_plots_control_state(self):
        self.setUp()

        self.visualiser.plot(self.sim)

    def test_close_removes_figure(self):
        self.setUp()
        self.visualiser.plot(self.sim)

        self.visualiser.close()

        self.assertIsNone(self.visualiser.figure)
        self.assertIsNone(self.visualiser.axes)


class TestFlightGearVisualiser(unittest.TestCase):
    env = None
    sim = None
    flightgear = None

    def setUp(self):
        if self.env:
            self.env.close()
        if self.sim:
            self.sim.close()
        self.task = BasicFlightTask()
        self.env = JsbSimEnv(task_type=BasicFlightTask)
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
        self.flightgear = FlightGearVisualiser(self.sim, self.task.get_props_to_output(),
                                               block_until_loaded=False)
        self.assertIsInstance(self.flightgear.figure, FigureVisualiser)

    def test_launch_flightgear(self):
        self.flightgear = FlightGearVisualiser(self.sim, self.task.get_props_to_output(),
                                               block_until_loaded=False)
        time.sleep(0.5)

        # check FlightGear has launched by looking at stdout
        self.assertIn('FlightGear', self.flightgear.flightgear_process.stdout.readline().decode())
        self.flightgear.close()

    def test_close_closes_flightgear(self):
        self.flightgear = FlightGearVisualiser(self.sim, self.task.get_props_to_output(),
                                               block_until_loaded=False)
        self.flightgear.close()
        timeout_seconds = 2.0
        return_code = self.flightgear.flightgear_process.wait(timeout=timeout_seconds)
        # a non-None return code indicates termination
        self.assertIsNotNone(return_code)

    def test_plot_displays_actions(self):
        self.setUp()
        self.flightgear = FlightGearVisualiser(self.sim, self.task.get_props_to_output(),
                                               block_until_loaded=False)
        self.flightgear.plot(self.sim)

        # the figure should have plotted a Lines object each axis
        for axis in ['axes_stick', 'axes_rudder', 'axes_throttle']:
            axis_data_plots = getattr(self.flightgear.figure.axes, axis)
            is_empty_plot = len(axis_data_plots.axes.lines) == 0
            self.assertFalse(is_empty_plot,
                             msg=f'no data plotted on axis {axis}')
