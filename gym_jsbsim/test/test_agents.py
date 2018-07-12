import unittest
import numpy as np
from gym_jsbsim.agents import RandomAgent, ConstantAgent, ConstantChangeNothingAgent
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


class TestConstantChangeNothingAgent(unittest.TestCase):
    def setUp(self, state_indices_for_actions=(0, 1, 0, 1)):
        self.task = TaskStub()
        self.action_space = self.task.get_action_space()
        self.state_space = self.task.get_observation_space()
        self.agent = ConstantChangeNothingAgent(action_space=self.action_space,
                                                state_indices_for_actions=state_indices_for_actions)

    def test_act_returns_correct_values(self):
        state_indices_for_actions = [2, 1, 0, 2]
        self.setUp(state_indices_for_actions=state_indices_for_actions)
        state = self.state_space.sample()
        action = self.agent.act(state)

        # action should match state values at specified indices
        expected_action = []
        for i in state_indices_for_actions:
            expected_action.append(state[i])
        expected_action = np.array(expected_action)

        np.testing.assert_array_equal(action, expected_action)