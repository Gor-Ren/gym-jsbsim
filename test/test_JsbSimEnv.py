import unittest
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from JsbSimEnv import JsbSimEnv

class TestJsbSimInstance(unittest.TestCase):

    def setUp(self):
        self.env = None
        self.env = JsbSimEnv()
        self.env.reset()

    def validate_observation(self, obs: np.array):
        """ Helper; checks shape and values of an observation. """
        self.assertEqual(self.env.observation_space.shape, obs.shape,
                         msg='observation has wrong size')
        self.assertTrue(self.env.observation_space.contains(obs),
                        msg=f'observation size or values out of range: {obs}\n'
                            f'expected min: {self.env.observation_space.low}\n'
                            f'expected max: {self.env.observation_space.high}')

    def validate_action(self, action: np.array):
        """ Helper; checks shape and values of an action. """
        self.assertEqual(self.env.action_space.shape, action.shape,
                         msg='action has wrong size')
        self.assertTrue(self.env.action_space.contains(action),
                        msg=f'action size or values out of range: {action}\n'
                            f'expected min: {self.env.action_space.low}\n'
                            f'expected max: {self.env.action_space.high}')

    def validate_action_made(self, action: np.array):
        """ Helper; confirms action was correctly input to simulation. """
        self.validate_action(action)
        for prop, command in zip(self.env.action_names, action):
            actual = self.env.sim[prop]
            self.assertAlmostEqual(command, actual,
                                   msg='simulation commanded value does not match action')

    def test_init_spaces(self):
        # check correct types for obs and action space
        self.assertIsInstance(self.env.observation_space, gym.Space,
                              msg='observation_space is not a Space object')
        self.assertIsInstance(self.env.action_space, gym.Space,
                              msg='action_space is not a Space object')

        # check low and high values are as expected
        obs_lows = self.env.observation_space.low
        obs_highs = self.env.observation_space.high
        act_lows = self.env.action_space.low
        act_highs = self.env.action_space.high

        places_tol = 3

        self.assertEqual('attitude/pitch-rad', self.env.observation_names[1])
        self.assertAlmostEqual(-0.5 * math.pi, obs_lows[1], places=places_tol,
                               msg='Pitch low range should be -pi/2')
        self.assertAlmostEqual(0.5 * math.pi, obs_highs[1], places=places_tol,
                               msg='Pitch high range should be +pi/2')
        self.assertEqual('attitude/roll-rad', self.env.observation_names[2])
        self.assertAlmostEqual(-1 * math.pi, obs_lows[2], places=places_tol,
                               msg='Roll low range should be -pi')
        self.assertAlmostEqual(1 * math.pi, obs_highs[2], places=places_tol,
                               msg='Roll high range should be +pi')

        self.assertEqual('fcs/aileron-cmd-norm', self.env.action_names[0])
        self.assertAlmostEqual(-1, act_lows[0], places=places_tol,
                               msg='Aileron command low range should be -1.0')
        self.assertAlmostEqual(1, act_highs[0], places=places_tol,
                               msg='Aileron command high range should be +1.0')
        self.assertEqual('fcs/throttle-cmd-norm', self.env.action_names[3])
        self.assertAlmostEqual(0, act_lows[3], places=places_tol,
                               msg='Throttle command low range should be 0.0')
        self.assertAlmostEqual(1, act_highs[3], places=places_tol,
                               msg='Throttle command high range should be +1.0')

    def test_reset_env(self):
        self.setUp()
        obs = self.env.reset()

        self.validate_observation(obs)

    def test_do_action(self):
        self.setUp()
        action1 = np.array([0.0, 0.0, 0.0, 0.0])
        action2 = np.array([-0.5, 0.9, -0.05, 0.75])

        # do an action and check results
        obs, _, _, _ = self.env.step(action1)
        self.validate_observation(obs)
        self.validate_action_made(action1)

        # repeat action several times
        for _ in range(10):
            obs, _, _, _ = self.env.step(action1)
            self.validate_observation(obs)
            self.validate_action_made(action1)

        # repeat new action
        for _ in range(10):
            obs, _, _, _ = self.env.step(action2)
            self.validate_observation(obs)
            self.validate_action_made(action2)

    def test_figure_created_closed(self):
        self.env.render(mode='human')
        self.assertIsInstance(self.env.figure, plt.Figure)
        self.env.close()
        self.assertIsNone(self.env.figure)

    def test_plot_positions(self):
        self.setUp()
        self.env.render(mode='human')

        xs = [50, 50.1, 50.2, 50.3, 50.4, 50.5]
        ys = [2, 2.5, 3.0, 3.5, 4.0, 4.5]
        zs = [1000, 950, 900, 850, 800]
        v_xs = [0, 100, 150, 200, 250]
        v_ys = [300, 250, 200, 150, 100]
        v_zs = [50, 50, 150, 200, 200]

        for x, y, z, v_x, v_y, v_z in zip(xs, ys, zs, v_xs, v_ys, v_zs):
            self.env._plot(x, y, z, v_x, v_y, v_z)
