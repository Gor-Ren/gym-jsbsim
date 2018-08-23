import unittest
import numpy as np
import itertools
import gym_jsbsim.properties as prp
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.tasks import HeadingControlTask, TurnHeadingControlTask
from gym_jsbsim.tests.stubs import FlightTaskStub
from typing import Iterable, Dict


class TestSteadyLevelFlightTask(unittest.TestCase):

    def setUp(self):
        self.class_under_test = self.get_class_under_test()
        self.task = self.class_under_test()

    def get_class_under_test(self):
        return HeadingControlTask

    def test_reward_calc(self):
        dummy_sim = FlightTaskStub({
            'velocities/h-dot-fps': 1,
            'attitude/roll-rad': -2,
        })
        expected_reward = 0
        for prop, _, gain in self.class_under_test.target_values:
            expected_reward -= abs(dummy_sim[prop]) * gain
        dummy_sim['position/h-sl-ft'] = 3000  # above minimum
        self.assertAlmostEqual(expected_reward, self.task._calculate_reward(dummy_sim))

        # test again with low altitude
        dummy_sim['position/h-sl-ft'] = 0
        expected_reward += self.task.TOO_LOW_REWARD
        self.assertAlmostEqual(expected_reward, self.task._calculate_reward(dummy_sim))

    def test_is_done_false(self):
        dummy_sim = FlightTaskStub({'simulation/sim-time-sec': 1,
                             'position/h-sl-ft': 5000})
        self.assertFalse(self.task._is_done(dummy_sim))

    def test_is_done_true_too_low(self):
        dummy_sim = FlightTaskStub({'simulation/sim-time-sec': 0,
                             'position/h-sl-ft': 0})
        self.assertTrue(self.task._is_done(dummy_sim))

    def test_is_done_true_time_out(self):
        dummy_sim = FlightTaskStub({'simulation/sim-time-sec': 9999,
                             'position/h-sl-ft': 5000})
        self.assertTrue(self.task._is_done(dummy_sim))

    def test_task_first_observation(self, custom_properties=None):
        props_value = 5
        if custom_properties is None:
            props = self.task.state_variables
        else:
            props = custom_properties
        dummy_sim = self.make_dummy_sim_with_all_props_set(props, props_value)
        state = self.task.observe_first_state(dummy_sim)

        number_of_state_vars = len(self.task.state_variables)
        expected_state = np.full(shape=(number_of_state_vars,), fill_value=5, dtype=int)

        self.assertIsInstance(state, np.ndarray)
        np.testing.assert_array_equal(expected_state, state)

        # check throttle and mixture set
        self.assertAlmostEqual(self.task.THROTTLE_CMD, dummy_sim[prp.throttle_cmd])
        self.assertAlmostEqual(self.task.MIXTURE_CMD, dummy_sim[prp.mixture_cmd])

    def make_dummy_sim_with_all_props_set(self, props: Iterable[prp.Property], value):
        """ Makes a DummySim, creates keys of 'name' from each property set to value """
        prop_names = (prop.name for prop in props)
        return FlightTaskStub(zip(prop_names, itertools.repeat(value)))

    def test_get_initial_conditions(self):
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

    def test_engines_init_running(self):
        env = JsbSimEnv(task_type=HeadingControlTask)

        # test assumption that property 'propulsion/engine/set-running'
        #   is zero prior to engine start!
        check_sim = Simulation(init_conditions={})
        engine_off_value = 0.0
        self.assertAlmostEqual(engine_off_value,
                               check_sim[prp.engine_running])
        check_sim.close()

        # check engines on once env has been reset
        _ = env.reset()
        engine_running_value = 1.0
        self.assertAlmostEqual(engine_running_value,
                               env.sim[prp.engine_running])

    def test_shaped_reward(self):
        low_reward_state_sim = FlightTaskStub.make_valid_state_stub(self.task)
        high_reward_state_sim = FlightTaskStub.make_valid_state_stub(self.task)

        # make one sim near the target values, and one relatively far away
        for prop, ideal_value, _ in self.task.target_values:
            low_reward_state_sim[prop] = ideal_value + 5
            high_reward_state_sim[prop] = ideal_value + 0.05
        # make sure altitude hasn't randomly been set below minimum!
        low_reward_state_sim[prp.altitude_sl_ft] = HeadingControlTask.MIN_ALT_FT + 1000
        high_reward_state_sim[prp.altitude_sl_ft] = HeadingControlTask.MIN_ALT_FT + 1000

        # suppose we start in the low reward state then transition to the high reward state
        self.task.observe_first_state(low_reward_state_sim)
        dummy_action = self.task.get_action_space().sample()
        _, first_reward, _, _ = self.task.task_step(high_reward_state_sim, dummy_action, 1)
        # shaped reward should be positive
        self.assertGreater(first_reward, 0)

        # now suppose we transition in the next step back to the low reward state
        _, second_reward, _, _ = self.task.task_step(low_reward_state_sim, dummy_action, 1)
        # shaped reward should be negative, and equal to the negative first_reward
        self.assertLess(second_reward, 0)
        self.assertAlmostEqual(-1 * first_reward, second_reward)

        # and if we remain in the same low-reward state we should receive 0 shaped reward
        _, third_reward, _, _ = self.task.task_step(low_reward_state_sim, dummy_action, 1)
        self.assertAlmostEqual(0, third_reward)


class TestHeadingControlTask(TestSteadyLevelFlightTask):
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
        dummy_sim = self.make_dummy_sim_with_all_props_set(self.task_state_property_dicts(), props_value)
        state = self.task.observe_first_state(dummy_sim)

        number_of_state_vars = len(self.task.state_variables)
        expected_state = np.full(shape=(number_of_state_vars,), fill_value=5, dtype=int)

        self.assertIsInstance(state, np.ndarray)
        np.testing.assert_array_equal(expected_state[:-1], state[:-1])  # last element holds random value

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
