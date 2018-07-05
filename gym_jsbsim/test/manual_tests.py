import unittest
import time
from gym_jsbsim.environment import JsbSimEnv
from typing import Type
from gym_jsbsim import tasks
from gym_jsbsim.test.stubs import TaskStub
from gym_jsbsim.agents import RandomAgent


class TestJsbSimInstance(unittest.TestCase):
    def setUp(self, task_type: Type[tasks.TaskModule]= TaskStub):
        self.env = None
        self.env = JsbSimEnv(task_type)
        self.env.reset()

    def test_long_episode_random_actions(self):
        self.setUp()
        tic = time.time()
        self.env.reset()
        for i in range(2000):
            self.env.step(action=self.env.action_space.sample())
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
                self.env.render(mode='human', action_names=self.env.task.action_names, action_values=action)

    def test_render_steady_level_flight_random(self):
        """ Runs steady level flight task with a random agent. """
        seed = 1
        self.setUp(task_type=tasks.SteadyLevelFlightTask)
        agent = RandomAgent(self.env.action_space, seed=seed)
        render_every = 5
        ep_reward = 0
        done = False
        state = self.env.reset()
        step_number = 0
        while not done:
            action = agent.act(state)
            state, reward, done, info = self.env.step(action)
            ep_reward += reward
            if step_number % render_every == 0:
                self.env.render(mode='human', action_names=self.env.task.action_names, action_values=action)
            step_number += 1


class FlightGearRenderTest(unittest.TestCase):
    def setUp(self, task_type: Type[tasks.TaskModule]=TaskStub):
        self.env = None
        self.env = JsbSimEnv(task_type)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_render_steady_level_flight_random(self):
        seed = 1
        self.setUp(task_type=tasks.SteadyLevelFlightTask)
        agent = RandomAgent(self.env.action_space, seed=seed)
        render_every = 5
        report_every = 20

        ep_reward = 0
        done = False
        state = self.env.reset()
        self.env.render(mode='flightgear')
        step_number = 0
        while not done:
            action = agent.act(state)
            state, reward, done, info = self.env.step(action)
            ep_reward += reward
            if step_number % render_every == 0:
                self.env.render(mode='flightgear', action_names=self.env.task.action_names, action_values=action)
                self.env.render(mode='human', action_names=self.env.task.action_names, action_values=action)
            if step_number % report_every == 0:
                print(f'time: {self.env.sim.get_sim_time()} s')
                print(f'last reward:\t{reward}')
                print(f'episode reward:\t{ep_reward}')
            step_number += 1
