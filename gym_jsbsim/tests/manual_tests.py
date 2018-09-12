import unittest
import time
import gym_jsbsim.properties as prp
from typing import Type
from gym_jsbsim import tasks, aircraft
from gym_jsbsim.tests.stubs import BasicFlightTask
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.agents import RandomAgent, ConstantAgent


class TestJsbSimInstance(unittest.TestCase):
    def setUp(self, task_type: Type[tasks.HeadingControlTask] = BasicFlightTask):
        self.env = None
        self.env = JsbSimEnv(task_type)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_long_episode_random_actions(self):
        self.setUp()
        tic = time.time()
        self.env.reset()
        for i in range(2000):
            self.env.step(action=self.env.action_space.sample())
            print(f'jsbsim {i / 10} s\n')
        toc = time.time()
        wall_time = (toc - tic)
        sim_time = self.env.sim.get_sim_time()
        print(f'Simulated {sim_time} s of flight in {wall_time} s')

    def test_render_episode(self):
        self.setUp()
        render_every = 5
        self.env.reset()
        for i in range(1000):
            action = self.env.action_space.sample()
            obs, _, _, _ = self.env.step(action=action)
            if i % render_every == 0:
                self.env.render(mode='human')

    def test_render_steady_level_flight_random(self):
        """ Runs steady level flight task with a random agent. """
        self.setUp(task_type=tasks.HeadingControlTask)
        agent = RandomAgent(self.env.action_space)
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
                self.env.render(mode='human')
            step_number += 1

    def test_run_episode_steady_level_flight_no_render(self):
        self.setUp(task_type=tasks.HeadingControlTask)
        agent = RandomAgent(self.env.action_space)
        report_every = 20
        EPISODES = 10

        for _ in range(EPISODES):
            ep_reward = 0
            done = False
            state = self.env.reset()
            step_number = 0
            while not done:
                action = agent.act(state)
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
                if step_number % report_every == 0:
                    print(f'time:\t{self.env.sim.get_sim_time()} s')
                    print(f'last reward:\t{reward}')
                    print(f'episode reward:\t{ep_reward}')
                step_number += 1


class FlightGearRenderTest(unittest.TestCase):
    def setUp(self, plane: aircraft.Aircraft = aircraft.cessna172P,
              task_type: Type[tasks.HeadingControlTask] = tasks.TurnHeadingControlTask):
        self.env = None
        self.env = JsbSimEnv(aircraft=plane, task_type=task_type)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_render_steady_level_flight(self):
        self.setUp(plane=aircraft.cessna172P, task_type=tasks.HeadingControlTask)
        agent = ConstantAgent(self.env.action_space)
        render_every = 5
        report_every = 20
        EPISODES = 999

        for _ in range(EPISODES):
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
                    self.env.render(mode='flightgear')
                if step_number % report_every == 0:
                    print(f'time:\t{self.env.sim.get_sim_time()} s')
                    print(f'last reward:\t{reward}')
                    print(f'episode reward:\t{ep_reward}')
                    print(f'thrust:\t{self.env.sim[prp.engine_thrust_lbs]}')
                    print(f'engine running:\t{self.env.sim[prp.engine_running]}')
                step_number += 1
            print(f'***\n'
                  f'EPISODE REWARD: {ep_reward}\n'
                  f'***')


class TurnHeadingControlTest(unittest.TestCase):
    def setUp(self, plane: aircraft.Aircraft = aircraft.cessna172P,
              task_type: Type[tasks.HeadingControlTask] = tasks.TurnHeadingControlTask,
              shaping: tasks.Shaping = tasks.Shaping.STANDARD):
        self.env = None
        self.env = JsbSimEnv(aircraft=plane, task_type=task_type, shaping=shaping)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_render_heading_control(self):
        self.setUp(plane=aircraft.a320, task_type=tasks.TurnHeadingControlTask,
                   shaping=tasks.Shaping.EXTRA_SEQUENTIAL)
        agent = RandomAgent(self.env.action_space)
        render_every = 5
        report_every = 20
        EPISODES = 50

        for _ in range(EPISODES):
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
                    self.env.render(mode='flightgear')
                if step_number % report_every == 0:
                    heading_target = tasks.HeadingControlTask.target_track_deg
                    print(f'time:\t{self.env.sim.get_sim_time()} s')
                    print(f'last reward:\t{reward}')
                    print(f'episode reward:\t{ep_reward}')
                    print(f'gear status:\t{self.env.sim[prp.gear]}')
                    print(f'thrust eng0:\t{self.env.sim[prp.engine_thrust_lbs]}')
                    print(f'thrust eng1:\t {self.env.sim[prp.Property("propulsion/engine[1]/thrust-lbs", "")]}')
                    print(f'heading:\t{self.env.sim[prp.heading_deg]}')
                    print(f'target heading:\t{self.env.sim[heading_target]}')
                    print('\n')
                step_number += 1
            print(f'***\n'
                  f'EPISODE REWARD: {ep_reward}\n'
                  f'***\n')
