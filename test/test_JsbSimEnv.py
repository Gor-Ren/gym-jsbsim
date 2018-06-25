import unittest
import gym
from JsbSimEnv import JsbSimEnv


class TestJsbSimWrapper(unittest.TestCase):

    def setUp(self):
        self.env = JsbSimEnv()

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

        self.assertEqual('attitude/pitch-rad', self.env.observation_names[0])
        self.assertAlmostEqual(-0.5, obs_lows[0],
                               msg='Pitch low range should be -0.5')
        self.assertAlmostEqual(0.5, obs_highs[0],
                               msg='Pitch high range should be +0.5')
        self.assertEqual('attitude/roll-rad', self.env.observation_names[1])
        self.assertAlmostEqual(-1, obs_lows[1],
                               msg='Roll low range should be -1.0')
        self.assertAlmostEqual(1, obs_highs[1],
                               msg='Roll high range should be +1.0')

        self.assertEqual('fcs/aileron-cmd-norm', self.env.action_names[0])
        self.assertAlmostEqual(-1, act_lows[0],
                               msg='Aileron command low range should be -1.0')
        self.assertAlmostEqual(1, act_highs[0],
                               msg='Aileron command high range should be +1.0')
        self.assertEqual('fcs/throttle-cmd-norm', self.env.action_names[3])
        self.assertAlmostEqual(0, act_lows[3],
                               msg='Throttle command low range should be 0.0')
        self.assertAlmostEqual(1, act_highs[3],
                               msg='Throttle command high range should be +1.0')