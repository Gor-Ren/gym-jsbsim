import unittest
import numpy as np
from tasks import SteadyLevelFlightTask
from test.stubs import SimStub


class TestSteadyLevelFlightTask(unittest.TestCase):
    def setUp(self):
        self.task = SteadyLevelFlightTask()

    def test_reward_calc(self):
        dummy_sim = SimStub({'accelerations/udot-ft_sec2': 1,
                             'accelerations/vdot-ft_sec2': 1,
                             'accelerations/wdot-ft_sec2': 1,
                             'accelerations/pdot-rad_sec2': -2,
                             'accelerations/qdot-rad_sec2': 2,
                             'accelerations/rdot-rad_sec2': 2,
                             'velocities/v-down-fps': 2,
                             'attitude/roll-rad': -2,
        })
        expected_reward = -sum(abs(val) ** 0.5 for val in dummy_sim.values())
        dummy_sim['position/h-sl-ft'] = 3000  # above minimum
        self.assertAlmostEqual(expected_reward, self.task._calculate_reward(dummy_sim))

        # test again with low altitude
        dummy_sim['position/h-sl-ft'] = 0
        expected_reward += self.task.TOO_LOW_REWARD
        self.assertAlmostEqual(expected_reward, self.task._calculate_reward(dummy_sim))

    def test_is_done_false(self):
        dummy_sim = SimStub({'simulation/sim-time-sec': 1,
                             'position/h-sl-ft': 5000})
        self.assertFalse(self.task._is_done(dummy_sim))

    def test_is_done_true_too_low(self):
        dummy_sim = SimStub({'simulation/sim-time-sec': 0,
                             'position/h-sl-ft': 0})
        self.assertTrue(self.task._is_done(dummy_sim))

    def test_is_done_true_time_out(self):
        dummy_sim = SimStub({'simulation/sim-time-sec': 9999,
                             'position/h-sl-ft': 5000})
        self.assertTrue(self.task._is_done(dummy_sim))
