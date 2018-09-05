import unittest
import numpy as np
import gym
from gym_jsbsim import utils
from gym_jsbsim.agents import RandomAgent
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import HeadingControlTask
import gym_jsbsim.properties as prp


class AgentEnvInteractionTest(unittest.TestCase):
    """ Tests for agents interacting with env. """

    def init_and_reset_env(self, env: JsbSimEnv):
        self.assertIsInstance(env.task, HeadingControlTask)

        # we interact at 5 Hz, so we expect the sim to run 12 timesteps per
        #   interaction since it runs at 120 Hz
        self.assertEqual(12, env.sim_steps_per_agent_step)

        # we init a random agent with a seed
        agent = RandomAgent(action_space=env.action_space)
        self.assertEqual(env.action_space, agent.action_space)

        # this task has an action space of three controls: aileron, elevator, rudder
        expected_num_actions = 3
        self.assertEqual(expected_num_actions, len(agent.action_space.low))
        # we see that the action space has the correct low and high range of +-1.0
        expect_low = np.array([-1.0] * expected_num_actions)
        expect_high = np.array([1.0] * expected_num_actions)
        np.testing.assert_array_almost_equal(expect_high, env.action_space.high)
        np.testing.assert_array_almost_equal(expect_low, env.action_space.low)

        # we reset the env and receive the first state; the env is now ready
        state = env.reset()
        self.assertEqual(len(env.observation_space.low), len(state))

        # we close the env and JSBSim closes with it
        env.close()
        self.assertIsNone(env.sim.jsbsim)

    def take_step_with_random_agent(self, env: JsbSimEnv):
        agent = RandomAgent(action_space=env.action_space)

        # we set up for a loop through one episode
        first_state = env.reset()

        # we take a single step
        action = agent.act(first_state)
        state, reward, done, info = env.step(action)

        # we see the state has changed
        self.assertEqual(first_state.shape, state.shape)
        self.assertTrue(np.any(np.not_equal(first_state, state)),
                        msg='state should have changed after simulation step')
        expected_time_step_size = env.sim_steps_per_agent_step / env.JSBSIM_DT_HZ
        self.assertAlmostEqual(expected_time_step_size, env.sim.get_sim_time())
        self.assertFalse(done, msg='episode is terminal after only a single step')

        # the aircraft engines are running, as per initial conditions
        self.assertNotAlmostEqual(env.sim[prp.engine_thrust_lbs], 0)

        env.close()

    def test_init_and_reset_all_envs(self):
        for env_id in utils.get_env_id_kwargs_map():
            env = gym.make(env_id)
            self.init_and_reset_env(env)

    def test_take_step_with_random_agent_all_envs(self):
        for env_id in utils.get_env_id_kwargs_map():
            env = gym.make(env_id)
            self.take_step_with_random_agent(env)
