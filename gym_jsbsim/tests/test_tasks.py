import unittest
import numpy as np
import gym_jsbsim.properties as prp
from gym_jsbsim.assessors import Assessor
from gym_jsbsim import rewards
from gym_jsbsim.tasks import HeadingControlTask, TurnHeadingControlTask
from gym_jsbsim.tests.stubs import SimStub
from typing import Dict


class TestHeadingControlTask(unittest.TestCase):
    default_shaping = HeadingControlTask.Shaping.OFF
    default_episode_time_s = 15.0
    default_step_frequency_hz = 5
    default_max_distance_m = 72.0 * default_episode_time_s  # Cessna high speed = 140 kn = 72 m/s

    def setUp(self):
        self.task = self.make_task()
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)  # causes task to init new-episode attributes

        self.dummy_action = np.asarray([0 for _ in range(len(self.task.action_variables))])

    def make_task(self,
                  shaping_type: HeadingControlTask.Shaping = default_shaping,
                  episode_time_s: float = default_episode_time_s,
                  step_frequency_hz: float = default_step_frequency_hz,
                  max_distance_m: float = default_max_distance_m) -> HeadingControlTask:
        return HeadingControlTask(shaping_type=shaping_type,
                                  episode_time_s=episode_time_s,
                                  step_frequency_hz=step_frequency_hz,
                                  max_distance_m=max_distance_m)

    def test_init_shaping_off(self):
        task = self.make_task(shaping_type=HeadingControlTask.Shaping.OFF)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(2, len(task.assessor.base_components))
        self.assertFalse(task.assessor.shaping_components)  # assert empty

    def test_init_shaping_basic(self):
        task = self.make_task(shaping_type=HeadingControlTask.Shaping.BASIC)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(2, len(task.assessor.base_components))
        self.assertEqual(1, len(task.assessor.shaping_components))

    def test_init_shaping_additive(self):
        task = self.make_task(shaping_type=HeadingControlTask.Shaping.ADDITIVE)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(2, len(task.assessor.base_components))
        self.assertEqual(3, len(task.assessor.shaping_components))

    def test_get_intial_conditions_correct_target_heading(self):
        self.setUp()

        ics = self.task.get_initial_conditions()
        initial_heading = ics[prp.initial_heading_deg]

        self.assertAlmostEqual(HeadingControlTask.INITIAL_HEADING_DEG, initial_heading)

    def test_get_initial_conditions_contains_all_props(self):
        ics = self.task.get_initial_conditions()

        self.assertIsInstance(ics, dict)
        for prop, value in self.task.base_initial_conditions.items():
            self.assertAlmostEqual(value, ics[prop])

        steady_level_task_ic_properties = [prp.initial_u_fps,
                                           prp.initial_v_fps,
                                           prp.initial_w_fps,
                                           prp.initial_p_radps,
                                           prp.initial_q_radps,
                                           prp.initial_r_radps,
                                           prp.initial_heading_deg
                                           ]
        for prop in steady_level_task_ic_properties:
            self.assertIn(prop, ics.keys(),
                          msg='expected HeadingControlTask to set value for'
                              f'property {prop} but not found in ICs')

    def test_observe_first_state(self):
        sim = SimStub.make_valid_state_stub(self.task)

        first_state = self.task.observe_first_state(sim)

        self.assertEqual(len(first_state), len(self.task.state_variables))
        self.assertIsInstance(first_state, np.ndarray)

    def test_task_step_correct_return_types(self):
        sim = SimStub.make_valid_state_stub(self.task)
        steps = 1

        state, reward, done, info = self.task.task_step(sim, self.dummy_action, steps)

        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), len(self.task.state_variables))

        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_task_step_returns_reward_in_info(self):
        sim = SimStub.make_valid_state_stub(self.task)
        steps = 1

        _, reward_scalar, _, info = self.task.task_step(sim, self.dummy_action, steps)
        reward_object = info['reward']

        self.assertIsInstance(reward_object, rewards.Reward)
        self.assertAlmostEqual(reward_object.reward(), reward_scalar)

    def test_task_step_non_terminal_time(self):
        sim = SimStub.make_valid_state_stub(self.task)
        non_terminal_time = self.default_episode_time_s - 1
        sim[prp.sim_time_s] = non_terminal_time
        steps = 1

        _, _, done, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertFalse(done)

    def test_task_step_terminal_exactly_max_time(self):
        sim = SimStub.make_valid_state_stub(self.task)
        terminal_time = self.default_episode_time_s
        sim[prp.sim_time_s] = terminal_time
        steps = 1

        _, _, done, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertTrue(done)

    def test_task_step_terminal_over_time(self):
        sim = SimStub.make_valid_state_stub(self.task)
        terminal_time = self.default_episode_time_s + 1
        sim[prp.sim_time_s] = terminal_time
        steps = 1

        _, _, done, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertTrue(done)


class TestTurnHeadingControlTask(TestHeadingControlTask):
    task_prop_names = (
        'position/h-sl-ft',
        'velocities/h-dot-fps',
        'attitude/roll-rad',
        'velocities/phidot-rad_sec',
        'attitude/psi-deg',
        'velocities/psidot-rad_sec',
        'velocities/thetadot-rad_sec',
        'target/heading-deg',
    )

    def get_class_under_test(self):
        return TurnHeadingControlTask

    def test_task_first_observation(self):
        props_value = 5
        dummy_sim = self.make_dummy_sim_with_all_props_set(self.task_state_property_dicts(),
                                                           props_value)
        state = self.task.observe_first_state(dummy_sim)

        number_of_state_vars = len(self.task.state_variables)
        expected_state = np.full(shape=(number_of_state_vars,), fill_value=5, dtype=int)

        self.assertIsInstance(state, np.ndarray)
        np.testing.assert_array_equal(expected_state[:-1],
                                      state[:-1])  # last element holds random value

        # check throttle and mixture set
        self.assertAlmostEqual(self.task.THROTTLE_CMD, dummy_sim['fcs/throttle-cmd-norm'])
        self.assertAlmostEqual(self.task.MIXTURE_CMD, dummy_sim['fcs/mixture-cmd-norm'])

    def test_observe_first_state_creates_desired_heading_in_expected_range(self):
        dummy_sim = self.make_dummy_sim_with_all_props_set(self.task_state_property_dicts(), 0)

        state = self.task.observe_first_state(dummy_sim)

        desired_heading = state[-1]
        self.assertGreaterEqual(desired_heading, 0)
        self.assertLessEqual(desired_heading, 360)

    def test_observe_first_state_changes_desired_heading(self):
        dummy_sim = self.make_dummy_sim_with_all_props_set(self.task_state_property_dicts(), 0)
        state = self.task.observe_first_state(dummy_sim)
        desired_heading = state[-1]

        new_episode_state = self.task.observe_first_state(dummy_sim)
        new_desired_heading = new_episode_state[-1]

        self.assertNotEqual(desired_heading, new_desired_heading)

    def task_state_property_dicts(self) -> Dict:
        extra_task_props = tuple({'name': prop_name} for prop_name in self.task_prop_names)
        return self.task.state_variables + extra_task_props
