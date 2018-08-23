import unittest
import numpy as np
from gym_jsbsim.agents import RandomAgent, ConstantAgent
from gym_jsbsim.tests.stubs import FlightTaskStub


class TestRandomAgent(unittest.TestCase):
    def setUp(self):
        self.action_space = FlightTaskStub().get_action_space()
        self.agent = RandomAgent(action_space=self.action_space)

    def test_act_generates_valid_actions(self):
        num_test_actions = 5
        for _ in range(num_test_actions):
            action = self.agent.act(None)
            self.assertTrue(self.action_space.contains(action))


class TestConstantAgent(unittest.TestCase):
    def setUp(self):
        self.task = FlightTaskStub()
        self.action_space = self.task.get_action_space()
        self.agent = ConstantAgent(action_space=self.action_space)

    def test_act_generates_valid_actions(self):
        num_test_actions = 3
        for _ in range(num_test_actions):
            action = self.agent.act(None)
            self.assertTrue(self.action_space.contains(action))

    def test_act_returns_same_action(self):
        num_test_actions = 5
        old_action = self.agent.act(None)
        for _ in range(num_test_actions):
            action = self.agent.act(None)
            np.testing.assert_array_almost_equal(old_action, action)
            old_action = action
