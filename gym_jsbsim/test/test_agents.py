import unittest
import numpy as np
from gym_jsbsim.agents import RandomAgent, ConstantAgent, HoldPositionAgent
from gym_jsbsim.test.stubs import TaskStub


class TestRandomAgent(unittest.TestCase):
    def setUp(self):
        self.action_space = TaskStub().get_action_space()
        self.agent = RandomAgent(action_space=self.action_space)

    def test_act_generates_valid_actions(self):
        num_test_actions = 5
        for _ in range(num_test_actions):
            action = self.agent.act(None)
            self.assertTrue(self.action_space.contains(action))


class TestConstantAgent(unittest.TestCase):
    def setUp(self):
        self.task = TaskStub()
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


class TestHoldPositionAgent(unittest.TestCase):
    def setUp(self):
        self.task = TaskStub()
        self.agent = HoldPositionAgent(action_space=self.task.get_action_space(),
                                       action_names=self.task.action_names,
                                       state_names=self.task.state_names)

    def test_agent_inits_correctly(self):
        indices = self.agent.state_indices_for_actions
        self.assertEqual(len(indices), len(self.task.get_action_space().low))