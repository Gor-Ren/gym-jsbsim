import unittest
import time
from JsbSimEnv import JsbSimEnv


class TestJsbSimInstance(unittest.TestCase):
    def setUp(self):
        self.env = None
        self.env = JsbSimEnv()
        self.env.reset()

    def test_long_episode_random_actions(self):
        self.setUp()
        tic = time.time()
        obs = self.env.reset()
        for i in range(2000):
            result = self.env.step(action=self.env.action_space.sample())
            print(f'sim {i / 10} s\n')
        toc = time.time()
        wall_time = (toc - tic)
        sim_time = self.env.sim['simulation/sim-time-sec']
        print(f'Simulated {sim_time} s of flight in {wall_time} s')

    def test_render_episode(self):
        self.setUp()
        render_every = 5
        self.env.reset()
        for i in range(1000):
            action = self.env.action_space.sample()
            obs, _, _, _ = self.env.step(action=action)
            if i % render_every == 0:
                self.env.render(mode='human', action_names=self.env.action_names, action_values=action)

