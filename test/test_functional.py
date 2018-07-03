import unittest
import numpy as np
from agents.random import RandomAgent
from JsbSimEnv import JsbSimEnv
from tasks import SteadyLevelFlightTask


class AgentEnvInteractionTest(unittest.TestCase):
    """ Unit and functional tests for agents interacting with env. """

    def test_random_agent_steady_level_task(self):
        # we create an environment with the steady level flight task
        agent_interaction_hz = 8
        env = JsbSimEnv(task_type=SteadyLevelFlightTask,
                        agent_interaction_freq=agent_interaction_hz)
        self.assertIsInstance(env.task, SteadyLevelFlightTask)

        # we interact at 8 Hz, so we expect the sim to run 15 timesteps per
        #   interaction since it runs at 120 Hz
        self.assertEqual(120, env.DT_HZ)
        self.assertEqual(15, env.sim_steps)

        # we init a random agent with a seed
        seed = 1
        agent = RandomAgent(action_space=env.action_space, seed=seed)

        # this task has an action space of three controls: aileron, elevator, rudder
        self.assertEqual(env.action_space, agent.action_space)
        self.assertEqual(3, len(agent.action_space.low))
        # we see that the action space has the correct low and high range of +-1.0
        expect_low = np.array([-1.0, -1.0, -1.0])
        expect_high = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(expect_high, env.action_space.high)
        np.testing.assert_array_almost_equal(expect_low, env.action_space.low)


